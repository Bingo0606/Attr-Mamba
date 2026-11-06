import pytest

torch = pytest.importorskip("torch")

from model.customized_model import (
    AttrMambaDecoder,
    SentenceSemanticCalibration,
    TokenLevelVisionLanguageBoundaryDelineation,
)


class IdentitySSM(torch.nn.Module):
    def forward(self, x):
        return x


def test_scm_returns_same_spatial_shape():
    module = SentenceSemanticCalibration(dim=8, ss2d=IdentitySSM())
    visual = torch.randn(2, 8, 8, 8)
    sentence = torch.randn(2, 8)
    out = module(visual, sentence)
    assert out.shape == visual.shape


def test_bdm_returns_visual_and_token_states():
    module = TokenLevelVisionLanguageBoundaryDelineation(
        dim=8, window_size=2, ss2d=IdentitySSM()
    )
    visual = torch.randn(2, 8, 8, 8)
    tokens = torch.randn(2, 8, 5)
    visual_out, token_out = module(visual, tokens)
    assert visual_out.shape == visual.shape
    assert token_out.shape == tokens.shape


def test_bdm_uses_all_windows_as_one_grid():
    module = TokenLevelVisionLanguageBoundaryDelineation(
        dim=8, window_size=2, ss2d=IdentitySSM()
    )
    visual = torch.randn(1, 8, 8, 8)
    tokens = torch.randn(1, 8, 5)
    visual_out, token_out = module(visual, tokens, return_grid=True)
    assert visual_out.shape == visual.shape
    assert token_out.shape == tokens.shape
    assert module.last_grid_shape == (1, 16, 9, 8)


def test_decoder_can_return_progressive_states():
    # This test only checks the public state contract. The full decoder is
    # instantiated with the lightweight configuration used by the repository.
    decoder = AttrMambaDecoder(
        dims=[8, 8, 8, 8],
        num_stages=1,
        return_intermediates=True,
        use_checkpoint=False,
        norm_layer="ln2d",
        text_dim=8,
        ss2d=IdentitySSM(),
    )
    visual = torch.randn(1, 8, 8, 8)
    tokens = torch.randn(1, 8, 5)
    sentence = tokens[:, :, 0]
    logits, states = decoder(visual, tokens, None, sentence)
    assert logits.shape[0] == visual.shape[0]
    assert len(states) == 1
    assert set(states["stage_1"]) == {"scm_visual", "bdm_visual", "text"}
