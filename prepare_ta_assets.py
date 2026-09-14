"""Prepare or verify the exact local T+A inference assets; no downloads."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "roadmap_b"))

from src.ta_assets import AssetIntegrityError, prepare_assets, verify_asset_root  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--baseline-root", help="Retained frozen Roadmap A snapshot")
    parser.add_argument("--b-root", help="Retained artifacts/roadmap_b directory")
    parser.add_argument("--c1-root", help="Retained artifacts/roadmap_c1_v2 directory")
    args = parser.parse_args()
    if not args.verify_only and not all((args.baseline_root, args.b_root, args.c1_root)):
        parser.error("Preparation requires --baseline-root, --b-root and --c1-root")
    try:
        result = (
            verify_asset_root(args.asset_root)
            if args.verify_only
            else prepare_assets(
                args.asset_root,
                baseline_root=args.baseline_root,
                b_root=args.b_root,
                c1_root=args.c1_root,
            )
        )
    except (AssetIntegrityError, OSError, ValueError) as exc:
        parser.exit(2, f"T+A asset error: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
