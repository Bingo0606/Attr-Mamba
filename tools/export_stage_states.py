"""Export saved Attr-Mamba stage states as compact heatmap images.

The evaluator can be run with ``--dump-stage-states`` and a small wrapper can
save the returned ``stage_states`` dictionary with ``torch.save``. This tool
does not rerun the model; it only converts those states into deterministic
visual summaries for Fig. 6-style inspection.
"""

import argparse
from pathlib import Path

import torch
from PIL import Image


def _heatmap(tensor):
    if tensor.dim() == 4:
        tensor = tensor[0]
    if tensor.dim() != 3:
        raise ValueError(f"expected [C,H,W] or [B,C,H,W], got {tuple(tensor.shape)}")
    response = tensor.detach().float().abs().mean(dim=0)
    response = response - response.min()
    scale = response.max().clamp_min(1e-8)
    response = (255.0 * response / scale).byte().cpu().numpy()
    return Image.fromarray(response, mode="L")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="torch file containing a stage_states dictionary")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    payload = torch.load(args.input, map_location="cpu")
    states = payload.get("stage_states", payload)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stage_name, state in sorted(states.items()):
        visual = state.get("bdm_visual", state.get("scm_visual"))
        if visual is None:
            continue
        _heatmap(visual).save(args.output_dir / f"{stage_name}_bdm_visual.png")
    print(f"Wrote stage heatmaps to {args.output_dir}")


if __name__ == "__main__":
    main()
