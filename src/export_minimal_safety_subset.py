import argparse
import json
from pathlib import Path

from loguru import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a minimal JSONL with prompt, response, and label only."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset/pku_saferlhf_pos_neg_100x2.jsonl"),
        help="Input JSONL path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/pku_saferlhf_pos_neg_100x2_minimal.jsonl"),
        help="Output JSONL path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with (
        args.input.open("r", encoding="utf-8") as infile,
        args.output.open("w", encoding="utf-8") as outfile,
    ):
        for line in infile:
            if not line.strip():
                continue
            record = json.loads(line)
            minimal = {
                "prompt": record["prompt"],
                "response": record["response"],
                "label": record["label"],
            }
            outfile.write(json.dumps(minimal, ensure_ascii=False) + "\n")
            count += 1

    logger.info("Exported {} records to {}", count, args.output)


if __name__ == "__main__":
    main()
