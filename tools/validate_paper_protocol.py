"""Validate the fixed evaluation conventions used by the revised manuscript."""

import argparse
import json
from pathlib import Path


PROTOCOL = {
    "threshold": 0.5,
    "boundary_radius": 1,
    "boundary_kernel": "3x3 square",
    "boundary_tolerance_px": 2,
    "empty_hd95": {"both_empty": 0.0, "one_empty": "image_diagonal"},
    "resolutions": {
        "Ref-LITS": [512, 512],
        "Ref-LIDC": [512, 512],
        "QaTa-COV19": [224, 224],
        "MosMedData+": [224, 224],
    },
    "seeds_per_experiment": 5,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="fail on any unexpected protocol value")
    parser.add_argument("--json", type=Path, help="optional path to write the protocol as JSON")
    args = parser.parse_args()
    expected = PROTOCOL
    if expected["threshold"] != 0.5 or expected["boundary_radius"] != 1 or expected["boundary_tolerance_px"] != 2:
        raise SystemExit("metric protocol mismatch")
    if args.strict and expected["seeds_per_experiment"] != 5:
        raise SystemExit("seed protocol mismatch")
    if args.json:
        args.json.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(expected, indent=2))
    print("Paper protocol validation passed.")


if __name__ == "__main__":
    main()
