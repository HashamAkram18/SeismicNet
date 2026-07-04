"""Generate train/val/test split files from STEAD metadata.

Reads the STEAD metadata CSV, splits by event ID (NOT by sample),
and writes data/splits/train.txt, val.txt, test.txt.

Split ratios: 70% train / 15% val / 15% test.

Rules:
- Split unit is event ID, not sample index
- The same event must not appear in more than one split
- No station should contribute more than 5% of a single split (warn if violated)
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def generate_splits(
    metadata_csv: Path,
    output_dir: Path,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> None:
    """Generate train/val/test split files from STEAD metadata.

    Args:
        metadata_csv: Path to STEAD metadata CSV file.
        output_dir: Directory to write split files.
        seed: Random seed for reproducibility.
        train_ratio: Fraction of events for training.
        val_ratio: Fraction of events for validation.
        test_ratio: Fraction of events for testing.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    # Load metadata
    logger.info("Loading metadata from %s", metadata_csv)
    df = pd.read_csv(metadata_csv, index_col=0)

    # Get unique event IDs
    event_ids = df.index.unique().tolist()
    n_events = len(event_ids)
    logger.info("Found %d unique events", n_events)

    # Shuffle event IDs
    rng = np.random.RandomState(seed)
    rng.shuffle(event_ids)

    # Compute split boundaries
    n_train = int(n_events * train_ratio)
    n_val = int(n_events * val_ratio)

    train_events = sorted(event_ids[:n_train])
    val_events = sorted(event_ids[n_train : n_train + n_val])
    test_events = sorted(event_ids[n_train + n_val :])

    logger.info(
        "Split sizes: train=%d (%.1f%%), val=%d (%.1f%%), test=%d (%.1f%%)",
        len(train_events), 100 * len(train_events) / n_events,
        len(val_events), 100 * len(val_events) / n_events,
        len(test_events), 100 * len(test_events) / n_events,
    )

    # Check for overlap
    all_events = set(train_events) | set(val_events) | set(test_events)
    assert len(all_events) == n_events, "Events lost during split"
    assert len(set(train_events) & set(val_events)) == 0, "Train/val overlap"
    assert len(set(train_events) & set(test_events)) == 0, "Train/test overlap"
    assert len(set(val_events) & set(test_events)) == 0, "Val/test overlap"

    # Check station distribution per split
    for split_name, events in [("train", train_events), ("val", val_events), ("test", test_events)]:
        split_df = df.loc[events]
        station_counts = split_df["receiver_code"].value_counts()
        total = len(split_df)
        max_station_pct = station_counts.iloc[0] / total * 100 if len(station_counts) > 0 else 0
        max_station = station_counts.index[0] if len(station_counts) > 0 else "N/A"

        if max_station_pct > 5.0:
            logger.warning(
                "Split '%s': station '%s' contributes %.1f%% (>5%% threshold)",
                split_name, max_station, max_station_pct,
            )
        else:
            logger.info(
                "Split '%s': max station contribution %.1f%% (station '%s')",
                split_name, max_station_pct, max_station,
            )

    # Write split files
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, events in [("train", train_events), ("val", val_events), ("test", test_events)]:
        split_path = output_dir / f"{split_name}.txt"
        split_path.write_text("\n".join(events) + "\n")
        logger.info("Wrote %s (%d events) to %s", split_name, len(events), split_path)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate train/val/test split files from STEAD metadata."
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("data/raw/metadata.csv"),
        help="Path to STEAD metadata CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits"),
        help="Output directory for split files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    generate_splits(
        metadata_csv=args.metadata_csv,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
