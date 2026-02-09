import argparse
import glob
import os

from loguru import logger

from plot_probe_utils import compute_cosine_similarity
from plot_probe_utils import find_probe_layers
from plot_probe_utils import load_probe_weights_for_steer_layer
from plot_probe_utils import parse_optional_layers
from plot_probe_utils import plot_cosine_heatmap
from utils import get_model_name_for_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot cosine similarity heatmaps for LDA probe weights"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="Model name used in assets path",
    )
    parser.add_argument(
        "--concepts",
        type=str,
        default="",
        help="Comma-separated concept names (defaults to all in directory)",
    )
    parser.add_argument(
        "--steer_layers",
        type=str,
        default="all",
        help="Comma-separated steer layers or 'all'",
    )
    parser.add_argument(
        "--probe_layers",
        type=str,
        default="all",
        help="Comma-separated probe layers or 'all'",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots/lda_probe_weights",
    )
    args = parser.parse_args()

    model_name = get_model_name_for_path(args.model)
    weight_root = os.path.join("assets", "lda_probe", model_name, "probe_weights")
    if not os.path.isdir(weight_root):
        raise FileNotFoundError(f"Probe weights directory not found: {weight_root}")

    if args.concepts:
        concepts = [c.strip() for c in args.concepts.split(",") if c.strip()]
    else:
        concepts = sorted(
            [
                name
                for name in os.listdir(weight_root)
                if os.path.isdir(os.path.join(weight_root, name))
            ]
        )

    if not concepts:
        logger.warning("No concept directories found in {}", weight_root)
        return

    steer_layers_filter = parse_optional_layers(args.steer_layers)
    probe_layers_filter = parse_optional_layers(args.probe_layers)

    for concept in concepts:
        concept_dir = os.path.join(weight_root, concept)
        steer_dirs = sorted(
            [
                d
                for d in glob.glob(os.path.join(concept_dir, "steer_*"))
                if os.path.isdir(d)
            ]
        )
        if not steer_dirs:
            logger.warning("No steer layer directories found for concept {}", concept)
            continue

        for steer_dir in steer_dirs:
            steer_name = os.path.basename(steer_dir)
            try:
                steer_layer = int(steer_name.replace("steer_", ""))
            except ValueError:
                logger.warning("Invalid steer layer directory: {}", steer_dir)
                continue
            if (
                steer_layers_filter is not None
                and steer_layer not in steer_layers_filter
            ):
                continue

            probe_layers = probe_layers_filter
            if probe_layers is None:
                probe_layers = find_probe_layers(steer_dir)

            if not probe_layers:
                logger.warning("No probe layers found for {} {}", concept, steer_name)
                continue

            for probe_layer in probe_layers:
                labels, weights = load_probe_weights_for_steer_layer(
                    steer_dir,
                    probe_layer,
                )
                if not labels:
                    logger.warning(
                        "No weights found for concept {} steer {} probe {}",
                        concept,
                        steer_layer,
                        probe_layer,
                    )
                    continue
                cosine = compute_cosine_similarity(weights)
                output_path = os.path.join(
                    args.output_dir,
                    model_name,
                    concept,
                    steer_name,
                    f"probe_{probe_layer}.png",
                )
                plot_cosine_heatmap(
                    cosine,
                    labels,
                    title=(
                        f"LDA probe weight cosine ({concept}) {steer_name} "
                        f"probe{probe_layer}"
                    ),
                    output_path=output_path,
                )


if __name__ == "__main__":
    main()
