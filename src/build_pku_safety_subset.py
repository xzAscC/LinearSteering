import argparse
import json
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset
from loguru import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a balanced PKU-SafeRLHF prompt-response dataset with "
            "harmless (pos) and harmful (neg) labels based on severity threshold."
        )
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="PKU-Alignment/PKU-SafeRLHF",
        help="Hugging Face dataset name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to load",
    )
    parser.add_argument(
        "--severity-threshold",
        type=int,
        default=0,
        help=(
            "Severity threshold: severity <= threshold => pos (harmless), "
            "severity > threshold => neg (harmful)."
        ),
    )
    parser.add_argument(
        "--num-per-class",
        type=int,
        default=100,
        help="Number of examples to sample for each class",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/pku_saferlhf_pos_neg_100x2.jsonl"),
        help="Output JSONL path",
    )
    return parser.parse_args()


def _record_from_row(
    row: dict[str, Any],
    row_index: int,
    response_id: int,
    threshold: int,
) -> dict[str, Any]:
    response_key = f"response_{response_id}"
    severity_key = f"response_{response_id}_severity_level"
    source_key = f"response_{response_id}_source"

    severity_level = int(row[severity_key])
    label = "pos" if severity_level <= threshold else "neg"

    return {
        "prompt": row["prompt"],
        "response": row[response_key],
        "label": label,
        "severity_level": severity_level,
        "row_index": row_index,
        "response_id": response_id,
        "prompt_source": row.get("prompt_source"),
        "response_source": row.get(source_key),
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    logger.info("Loading dataset: {} [{}]", args.dataset_name, args.split)
    dataset = load_dataset(args.dataset_name, split=args.split)

    pos_pool: list[dict[str, Any]] = []
    neg_pool: list[dict[str, Any]] = []

    for row_index, row in enumerate(dataset):
        for response_id in (0, 1):
            record = _record_from_row(
                row=row,
                row_index=row_index,
                response_id=response_id,
                threshold=args.severity_threshold,
            )
            if record["label"] == "pos":
                pos_pool.append(record)
            else:
                neg_pool.append(record)

    logger.info("Collected pos={} neg={}", len(pos_pool), len(neg_pool))

    if len(pos_pool) < args.num_per_class:
        raise ValueError(
            f"Not enough pos examples: need {args.num_per_class}, got {len(pos_pool)}"
        )
    if len(neg_pool) < args.num_per_class:
        raise ValueError(
            f"Not enough neg examples: need {args.num_per_class}, got {len(neg_pool)}"
        )

    sampled_pos = rng.sample(pos_pool, args.num_per_class)
    sampled_neg = rng.sample(neg_pool, args.num_per_class)

    sampled_records = sampled_pos + sampled_neg
    rng.shuffle(sampled_records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for record in sampled_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Saved {} records to {}", len(sampled_records), args.output)
    logger.info(
        "Config: threshold={} num_per_class={} seed={}",
        args.severity_threshold,
        args.num_per_class,
        args.seed,
    )


if __name__ == "__main__":
    main()
