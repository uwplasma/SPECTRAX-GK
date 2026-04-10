#!/usr/bin/env python3
"""Print the parsed contract of a GX reduced-model benchmark input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spectraxgk.gx_reduced_models import load_reduced_model_contract


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gx_input", type=Path, help="GX reduced-model input file (e.g. cetg.in)")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of key=value lines.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    contract = load_reduced_model_contract(args.gx_input)
    payload = contract.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    for key, value in payload.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
