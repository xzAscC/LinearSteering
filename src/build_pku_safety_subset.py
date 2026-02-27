import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
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
    parser.add_argument(
        "--plot-umap",
        action="store_true",
        help="Save a UMAP scatter plot for sampled records",
    )
    parser.add_argument(
        "--umap-output",
        type=Path,
        default=Path("plots/pku_saferlhf_pos_neg_100x2_umap.pdf"),
        help="UMAP plot output path",
    )
    parser.add_argument(
        "--umap-features",
        type=int,
        default=2048,
        help="Hashing feature dimension for text vectors",
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


def _stable_hash_token(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _texts_to_hashed_features(texts: list[str], feature_dim: int) -> np.ndarray:
    features = np.zeros((len(texts), feature_dim), dtype=np.float32)
    for row_idx, text in enumerate(texts):
        tokens = text.lower().split()
        if not tokens:
            continue
        for token in tokens:
            col = _stable_hash_token(token) % feature_dim
            features[row_idx, col] += 1.0
        norm = np.linalg.norm(features[row_idx])
        if norm > 0.0:
            features[row_idx] /= norm
    return features


def _plot_umap(
    records: list[dict[str, Any]], output_path: Path, seed: int, feature_dim: int
) -> None:
    if len(records) < 2:
        logger.warning("Skip UMAP plot: not enough records ({})", len(records))
        return

    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "UMAP plotting requires 'umap-learn'. Install it with: uv add umap-learn"
        ) from exc

    texts = [f"{record['prompt']}\n{record['response']}" for record in records]
    labels = [record["label"] for record in records]

    features = _texts_to_hashed_features(texts, feature_dim)
    n_neighbors = min(15, len(records) - 1)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=seed,
    )
    embedding = reducer.fit_transform(features)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))

    label_to_color = {"pos": "#1f77b4", "neg": "#d62728"}
    for label in ("pos", "neg"):
        indices = [idx for idx, y in enumerate(labels) if y == label]
        if not indices:
            continue
        points = embedding[indices]
        plt.scatter(
            points[:, 0],
            points[:, 1],
            s=24,
            alpha=0.8,
            c=label_to_color[label],
            label=label,
        )

    plt.title("UMAP of PKU-SafeRLHF Sampled Prompt-Response Pairs")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved UMAP plot to {}", output_path)


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
    if args.plot_umap:
        _plot_umap(
            records=sampled_records,
            output_path=args.umap_output,
            seed=args.seed,
            feature_dim=args.umap_features,
        )
    logger.info(
        "Config: threshold={} num_per_class={} seed={}",
        args.severity_threshold,
        args.num_per_class,
        args.seed,
    )


if __name__ == "__main__":
    main()
