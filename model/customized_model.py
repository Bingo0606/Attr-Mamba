import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from timm.models.registry import register_model

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


class _ChannelLayerNorm2d(nn.Module):
    """Dependency-light equivalent of VMamba's channels-first LayerNorm2d."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(True),
    )


def _make_ss2d(dim, kwargs):
    """Create the VMamba SS2D operator used by both paper modules."""
    from vmamba_model.vmamba import SS2D
    act_layer = kwargs.get("ssm_act_layer", nn.SiLU)
    if isinstance(act_layer, str):
        act_layer = {"silu": nn.SiLU, "gelu": nn.GELU, "relu": nn.ReLU}.get(act_layer.lower(), nn.SiLU)
    return SS2D(
        d_model=dim,
        d_state=kwargs.get("ssm_d_state", 16),
        ssm_ratio=kwargs.get("ssm_ratio", 2.0),
        dt_rank=kwargs.get("ssm_dt_rank", "auto"),
        act_layer=act_layer,
        d_conv=kwargs.get("ssm_conv", 3),
        conv_bias=kwargs.get("ssm_conv_bias", True),
        dropout=kwargs.get("ssm_drop_rate", 0.0),
        initialize=kwargs.get("ssm_init", "v0"),
        forward_type=kwargs.get("forward_type", "v052d3"),
        channel_first=True,
    )


class SentenceSemanticCalibration(nn.Module):
    """Sentence-level SCM from Eq. (1)--(3) of the manuscript."""

    def __init__(self, dim, ss2d=None, **kwargs):
        super().__init__()
        self.norm = _ChannelLayerNorm2d(dim)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim))
        self.ss2d = ss2d if ss2d is not None else _make_ss2d(dim, kwargs)

    def forward(self, visual, sentence):
        alpha, beta, gamma = self.modulation(sentence).chunk(3, dim=-1)
        calibrated = (1.0 + gamma[..., None, None]) * self.norm(visual)
        calibrated = calibrated + beta[..., None, None]
        return visual + alpha[..., None, None] * self.ss2d(calibrated)


class TokenLevelVisionLanguageBoundaryDelineation(nn.Module):
    """BDM with the paper's global P x (K^2+L) interaction grid.

    ``visual`` is [B,C,H,W] and ``text`` is [B,C,L]. All windows are
    stacked before SS2D, so the scan can propagate information across
    windows as described in the manuscript.
    """

    def __init__(self, dim, window_size=4, ss2d=None, **kwargs):
        super().__init__()
        self.window_size = window_size
        self.ss2d = ss2d if ss2d is not None else _make_ss2d(dim, kwargs)
        self.window_weights = nn.Parameter(torch.zeros(16))
        self.last_grid_shape = None

    def forward(self, visual, text, return_grid=False):
        if text.dim() == 4:
            text = text.mean(dim=-1)
        b, c, h, w = visual.shape
        k = self.window_size
        pad_h, pad_w = (k - h % k) % k, (k - w % k) % k
        padded = F.pad(visual, (0, pad_w, 0, pad_h))
        hp, wp = padded.shape[-2:]
        nh, nw = hp // k, wp // k

        windows = padded.view(b, c, nh, k, nw, k).permute(0, 2, 4, 1, 3, 5)
        windows = windows.reshape(b, nh * nw, c, k * k).permute(0, 1, 3, 2)
        p = windows.shape[1]
        tokens = text.transpose(1, 2).unsqueeze(1).expand(-1, p, -1, -1)
        grid = torch.cat((windows, tokens), dim=2).permute(0, 3, 1, 2).contiguous()
        self.last_grid_shape = (b, p, k * k + text.shape[-1], c)

        mixed = self.ss2d(grid).permute(0, 2, 3, 1)
        visual_out, text_out = mixed[:, :, : k * k], mixed[:, :, k * k :]
        visual_out = visual_out.permute(0, 1, 3, 2).reshape(b, nh, nw, c, k, k)
        visual_out = visual_out.permute(0, 3, 1, 4, 2, 5).reshape(b, c, hp, wp)
        visual_out = (visual_out + padded)[:, :, :h, :w]

        weights = self.window_weights
        if weights.numel() != p:
            weights = F.interpolate(
                weights[None, None], size=p, mode="linear", align_corners=False
            ).flatten()
        weights = torch.softmax(weights, dim=0)
        text_out = (text_out * weights[None, :, None, None]).sum(dim=1).transpose(1, 2)
        # The grid is intentionally kept internal; ``last_grid_shape`` makes
        # the interaction scope inspectable without changing the module API.
        return visual_out, text_out


# Kept as an import alias for downstream users; the implementation is now BDM.
MorphologyAwareBoundaryDelineation = TokenLevelVisionLanguageBoundaryDelineation


class AttrMambaDecoder(nn.Module):
    """Cascaded decoder implementing SCM -> BDM at each decoding stage."""

    def __init__(
        self,
        num_stages=4,
        disable_scm=False,
        disable_bdm=False,
        return_intermediates=False,
        **kwargs,
    ):
        super().__init__()
        if not 1 <= num_stages <= 6:
            raise ValueError("num_stages must be between 1 and 6")
        config_dims = list(kwargs.get("dims", [96]))
        if len(config_dims) >= 4 and len(set(config_dims)) > 1:
            stage_dims = list(reversed(config_dims[-4:]))
            stage_dims += [stage_dims[-1], stage_dims[-1]]
        else:
            stage_dims = [config_dims[0]] * 6
        self.stage_dims = stage_dims[:num_stages]
        self.num_stages = num_stages
        self.disable_scm = disable_scm
        self.disable_bdm = disable_bdm
        self.return_intermediates = return_intermediates
        self.text_dim = kwargs.get("text_dim", 768)

        self.image_guidance = nn.ModuleList()
        self.text_guidance = nn.ModuleList()
        self.scm = nn.ModuleList()
        self.bdm = nn.ModuleList()
        for i, dim in enumerate(self.stage_dims):
            in_dim = self.stage_dims[i - 1] if i else dim
            text_in = self.text_dim if i == 0 else self.stage_dims[i - 1]
            self.image_guidance.append(nn.Sequential(nn.Conv2d(in_dim, dim, 1), nn.ReLU(inplace=True)))
            self.text_guidance.append(nn.Sequential(nn.Linear(text_in, dim), nn.ReLU(inplace=True)))
            self.scm.append(SentenceSemanticCalibration(dim, **kwargs))
            self.bdm.append(TokenLevelVisionLanguageBoundaryDelineation(dim, window_size=4, **kwargs))

        final_dim = self.stage_dims[-1]
        self.proj_out = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            conv_layer(final_dim, max(final_dim // 2, 1), 3, padding=1),
            nn.Conv2d(max(final_dim // 2, 1), 1, 3, padding=1),
        )

    def forward(self, x, l_feat, l_mask=None, pooler_out=None, return_intermediates=None):
        if l_feat.dim() != 3:
            raise ValueError("l_feat must have shape [B,C,L]")
        tokens = l_feat
        sentence = pooler_out if pooler_out is not None else tokens[:, :, 0]
        states = {}
        for i in range(self.num_stages):
            x = self.image_guidance[i](x)
            sentence_i = self.text_guidance[i](sentence)
            tokens_i = self.text_guidance[i](tokens.transpose(1, 2)).transpose(1, 2)
            scm_visual = x if self.disable_scm else self.scm[i](x, sentence_i)
            if self.disable_bdm:
                bdm_visual, next_tokens = scm_visual, tokens_i
            else:
                bdm_visual, next_tokens = self.bdm[i](scm_visual, tokens_i)
            x, tokens = bdm_visual, next_tokens
            sentence = tokens.mean(dim=-1)
            states[f"stage_{i + 1}"] = {
                "scm_visual": scm_visual,
                "bdm_visual": bdm_visual,
                "text": tokens,
            }
        out = self.proj_out(x)
        want_states = self.return_intermediates if return_intermediates is None else return_intermediates
        return (out, states) if want_states else out


def _build_attr_mamba(model_size="tiny", **kwargs):
    from .segmenter import BaseSegmenter
    from .utils import update_mamba_config
    from .swin_image_encoder import MultiModalSwinTransformer

    config_dict = update_mamba_config(model_size)
    decoder_kwargs = {**config_dict, **kwargs}
    decoder = AttrMambaDecoder(**decoder_kwargs)

    if model_size == "base":
        embed_dim = 128
        depths = [2, 2, 18, 2]
        num_heads = [4, 8, 16, 32]
        window_size = 12
        mha = [8, 8, 8, 8]
        drop_path_rate = 0.3
    else:
        embed_dim = 96
        depths = [2, 2, 6, 2]
        num_heads = [3, 6, 12, 24]
        window_size = 7
        mha = [4, 4, 4, 4]
        drop_path_rate = 0.2

    backbone = MultiModalSwinTransformer(
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        ape=False,
        drop_path_rate=drop_path_rate,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=True,
        num_heads_fusion=mha,
        fusion_drop=0,
    )
    pretrained = kwargs.get("swin_pretrained", "")
    if pretrained and os.path.exists(pretrained):
        backbone.init_weights(pretrained=pretrained)
    elif pretrained:
        print(f"Warning: Swin pretrained weights not found: {pretrained}. Initializing backbone weights.")
        backbone.init_weights(pretrained=None)
    else:
        backbone.init_weights(pretrained=None)

    segmenter_kwargs = {**config_dict, **kwargs}
    return BaseSegmenter(backbone, decoder, **segmenter_kwargs)


@register_model
def AttrMamba(model_size="tiny", **kwargs):
    return _build_attr_mamba(model_size=model_size, **kwargs)
