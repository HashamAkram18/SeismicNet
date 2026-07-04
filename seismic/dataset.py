"""PyTorch Dataset for reading STEAD HDF5 files.

Handles loading waveforms and labels from the Stanford Earthquake Dataset,
applying preprocessing, and providing samples for training and evaluation.

Dataset split is by event ID, never by sample index. The same event must not
appear in more than one split file.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

from seismic.augmentation import augment_waveform
from seismic.preprocessing import preprocess_waveform

logger = logging.getLogger(__name__)

# Risk label derivation table (docs/01_project_overview.md § label_schema)
# Columns: depth < 20km, depth >= 20km
RISK_TABLE = {
    "low":      (0, 0),  # Ml < 3.0: always low
    "moderate": (1, 0),  # 3.0 <= Ml < 4.5: moderate/shallow, low/deep
    "high":     (2, 1),  # 4.5 <= Ml < 6.0: high/shallow, moderate/deep
    "critical": (3, 2),  # Ml >= 6.0: critical/shallow, high/deep
}


def derive_risk_label(magnitude: float, depth: float) -> int:
    """Derive structural damage risk class from magnitude and depth.

    Rule (docs/01_project_overview.md § label_schema):
        Ml < 3.0:           always low (0)
        3.0 <= Ml < 4.5:    moderate (1) if depth < 20km, else low (0)
        4.5 <= Ml < 6.0:    high (2) if depth < 20km, else moderate (1)
        Ml >= 6.0:          critical (3) if depth < 20km, else high (2)

    Args:
        magnitude: Local magnitude (Ml).
        depth: Event depth in km.

    Returns:
        Risk class integer (0=low, 1=moderate, 2=high, 3=critical).
    """
    if magnitude < 3.0:
        return 0  # low
    elif magnitude < 4.5:
        return 1 if depth < 20.0 else 0  # moderate or low
    elif magnitude < 6.0:
        return 2 if depth < 20.0 else 1  # high or moderate
    else:
        return 3 if depth < 20.0 else 2  # critical or high


def load_event_ids(split_path: Path) -> list[str]:
    """Load event IDs from a split file (one per line).

    Args:
        split_path: Path to train.txt, val.txt, or test.txt.

    Returns:
        List of event ID strings.
    """
    text = split_path.read_text()
    return [line.strip() for line in text.strip().splitlines() if line.strip()]


class SeismicDataset(Dataset):
    """Dataset for STEAD HDF5 seismic waveforms.

    Loads waveforms lazily from HDF5 files and metadata from a CSV file.
    The HDF5 file handle is opened per __getitem__ call (thread-safe) and
    closed immediately after reading.

    Args:
        hdf5_paths: Paths to STEAD HDF5 files (e.g., [chunk1.hdf5, chunk2.hdf5]).
        metadata_csv: Path to STEAD metadata CSV file.
        split_file: Path to split file (train.txt, val.txt, or test.txt).
        preprocessing_config: Preprocessing configuration dict.
        label_schema_config: Label schema configuration dict (reserved for future use).
        augment: Whether to apply augmentation (only True for training split).
    """

    def __init__(
        self,
        hdf5_paths: list[str],
        metadata_csv: str,
        split_file: str,
        preprocessing_config: dict[str, Any],
        label_schema_config: dict[str, Any] | None = None,
        augment: bool = False,
    ) -> None:
        self.hdf5_paths = [Path(p) for p in hdf5_paths]
        self.preprocessing_config = preprocessing_config
        self.augment = augment

        # Load event IDs from split file
        self.event_ids = load_event_ids(Path(split_file))

        # Load metadata CSV and index by event ID
        self.metadata = pd.read_csv(metadata_csv, index_col=0)

        # Validate all event IDs exist in metadata
        missing = set(self.event_ids) - set(self.metadata.index)
        if missing:
            raise ValueError(
                f"{len(missing)} event IDs in split file not found in metadata CSV"
            )

        # Build a mapping from event ID to HDF5 file path
        # Each HDF5 file contains a 'data' group with datasets keyed by event ID
        self._event_to_hdf5: dict[str, Path] = {}
        for hdf5_path in self.hdf5_paths:
            with h5py.File(hdf5_path, "r") as f:
                for event_id in f["data"]:
                    self._event_to_hdf5[event_id] = hdf5_path

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.event_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a single sample dict.

        Pipeline:
        1. Look up event ID from index
        2. Load waveform [3, 6000] from HDF5 (lazy open, close immediately)
        3. Load metadata row from CSV for this event ID
        4. Call preprocess_waveform() → get waveform [3, 3000] and pgv scalar
        5. If augment=True, call augmentation.augment_waveform()
        6. Derive Task D risk label from magnitude + depth
        7. Compute normalized S-pick position
        8. Return full dict with all 8 keys

        Returns:
            Dict with keys:
                "waveform": Tensor[3, 3000]
                "pgv": Tensor[1]
                "label_detection": Tensor[1] (0.0 or 1.0)
                "label_phase_pick": Tensor[2] ([p_norm, s_norm])
                "label_magnitude": Tensor[1] (Ml value, NaN for noise)
                "label_risk": Tensor[1] (long, class index 0–3, -1 for noise)
                "is_noise": Tensor[1] (bool)
                "event_id": str
        """
        event_id = self.event_ids[idx]

        # Load waveform from HDF5 (lazy open, close immediately for thread safety)
        hdf5_path = self._event_to_hdf5[event_id]
        with h5py.File(hdf5_path, "r") as f:
            waveform: NDArray[np.floating] = f["data"][event_id]["waveforms"][:]

        # Load metadata row
        meta = self.metadata.loc[event_id]

        # Extract metadata fields
        p_arrival_sample = int(meta["p_arrival_sample"])
        s_arrival_sample = int(meta["s_arrival_sample"])
        magnitude = float(meta["source_magnitude"])
        depth = float(meta["source_depth_km"])
        trace_category = str(meta["trace_category"])

        # Determine if this is a noise window
        is_noise = trace_category == "noise"

        # Step 4: Call preprocess_waveform
        result = preprocess_waveform(
            waveform=waveform,
            sampling_rate=self.preprocessing_config["sampling_rate_hz"],
            config=self.preprocessing_config,
            p_arrival_sample=p_arrival_sample,
            extract_pgv_flag=True,
        )

        processed_waveform = result["waveform"]  # [3, 3000] float32
        pgv = float(result["pgv"])

        # Step 5: Augmentation (only during training)
        if self.augment and not is_noise:
            processed_waveform, pgv = augment_waveform(processed_waveform, pgv)

        # Step 6: Derive Task D risk label
        if is_noise:
            risk_label = -1
            magnitude_value = float("nan")
        else:
            risk_label = derive_risk_label(magnitude, depth)
            magnitude_value = magnitude

        # Step 7: Compute normalized phase pick targets
        # P-pick is always at p_arrival_sample_index (1000) in the extracted window
        p_pick_normalized = self.preprocessing_config["p_arrival_sample_index"] / self.preprocessing_config["window_length_samples"]

        # S-pick: S relative to the extracted window start
        window_start = p_arrival_sample - self.preprocessing_config["p_arrival_sample_index"]
        s_pick_raw = (s_arrival_sample - window_start) / self.preprocessing_config["window_length_samples"]
        s_pick_normalized = float(np.clip(s_pick_raw, 0.0, 1.0))

        # Step 8: Build return dict
        return {
            "waveform": torch.from_numpy(processed_waveform).float(),
            "pgv": torch.tensor([pgv], dtype=torch.float32),
            "label_detection": torch.tensor([0.0 if is_noise else 1.0], dtype=torch.float32),
            "label_phase_pick": torch.tensor([p_pick_normalized, s_pick_normalized], dtype=torch.float32),
            "label_magnitude": torch.tensor([magnitude_value], dtype=torch.float32),
            "label_risk": torch.tensor([risk_label], dtype=torch.long),
            "is_noise": torch.tensor([is_noise], dtype=torch.bool),
            "event_id": event_id,
        }


def get_dataloader(
    hdf5_paths: list[str],
    metadata_csv: str,
    split_file: str,
    preprocessing_config: dict[str, Any],
    augment: bool = False,
    batch_size: int = 256,
    num_workers: int = 8,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    drop_last: bool = True,
) -> DataLoader:
    """Factory function that returns a configured DataLoader.

    Configuration from docs/02_data_pipeline.md § DataLoader configuration.

    Args:
        hdf5_paths: Paths to STEAD HDF5 files.
        metadata_csv: Path to STEAD metadata CSV.
        split_file: Path to split file.
        preprocessing_config: Preprocessing configuration dict.
        augment: Whether to apply augmentation.
        batch_size: Batch size (default 256).
        num_workers: Number of DataLoader workers (default 8).
        pin_memory: Pin memory for GPU transfer (default True).
        prefetch_factor: Prefetch factor (default 2).
        drop_last: Drop last incomplete batch (default True).

    Returns:
        Configured DataLoader instance.
    """
    dataset = SeismicDataset(
        hdf5_paths=hdf5_paths,
        metadata_csv=metadata_csv,
        split_file=split_file,
        preprocessing_config=preprocessing_config,
        augment=augment,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=augment,  # shuffle only for training
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        drop_last=drop_last,
    )
