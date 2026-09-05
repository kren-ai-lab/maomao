from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def find_repo_root(start=None) -> Path:
    configured = start or os.environ.get("MAOMAO_ROOT") or Path.cwd()
    start_path = Path(configured).expanduser().resolve()
    for candidate in (start_path, *start_path.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "maomao").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not locate the MAOMAO repository root from {start_path}. "
        "Run this script inside the repository or set MAOMAO_ROOT."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract one MAOMAO sequence card as JSON and HTML.",
    )
    query = parser.add_mutually_exclusive_group(required=True)
    query.add_argument("--id", dest="identifier", help="Complete sha256_<digest> identifier.")
    query.add_argument("--sequence", help="Canonical peptide sequence; whitespace and case are normalized.")
    parser.add_argument(
        "--profiles-dir",
        type=Path,
        help="Sequence-profile directory. Defaults to <repo>/sequence_profiles.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination directory. Defaults to <profiles-dir>/selected_cards.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root()
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from maomao.sequence_cards import export_sequence_card

    profiles_dir = (
        args.profiles_dir.expanduser().resolve()
        if args.profiles_dir is not None
        else repo_root / "sequence_profiles"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else profiles_dir / "selected_cards"
    )
    outputs = export_sequence_card(
        profiles_dir,
        output_dir,
        identifier=args.identifier,
        sequence=args.sequence,
    )

    print("Card found and exported:")
    print("JSON:", outputs["json"])
    print("HTML:", outputs["html"])


if __name__ == "__main__":
    main()
