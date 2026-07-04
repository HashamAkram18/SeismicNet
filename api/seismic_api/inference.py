"""Inference engine — loads exported model and runs prediction.

Singleton pattern: get_engine() returns the same instance across requests.
Loaded once at Django startup via AppConfig.ready().

CRITICAL: All signal processing calls seismic.preprocessing.preprocess_waveform().
This is Invariant 1 — never reimplement preprocessing here.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Module-level singleton
_engine: Optional["SeismicInferenceEngine"] = None
_lock = threading.Lock()


class SeismicInferenceEngine:
    """Loads exported TorchScript model and runs inference.

    The engine is created once at startup and reused for all requests.
    It holds the model, configs, and label schema in memory.
    """

    def __init__(self, artifact_dir: str) -> None:
        """Load model and configs from artifact directory.

        Args:
            artifact_dir: Path to exported model directory containing
                model_scripted.pt, preprocessing_config.json, model_config.json,
                label_schema.json.

        Raises:
            FileNotFoundError: If required artifact files are missing.
            ValueError: If preprocessing_config schema_version mismatch.
        """
        artifact_path = Path(artifact_dir)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact directory not found: {artifact_dir}")

        # Load preprocessing config
        import json

        preprocess_config_path = artifact_path / "preprocessing_config.json"
        if not preprocess_config_path.exists():
            raise FileNotFoundError(f"Missing preprocessing_config.json in {artifact_dir}")
        with open(preprocess_config_path) as f:
            self._preprocessing_config = json.load(f)
        self._preprocessing_config_version = self._preprocessing_config["schema_version"]

        # Load model config
        model_config_path = artifact_path / "model_config.json"
        if not model_config_path.exists():
            raise FileNotFoundError(f"Missing model_config.json in {artifact_dir}")
        with open(model_config_path) as f:
            self._model_config = json.load(f)
        self._detection_threshold = self._model_config.get("detection_threshold", 0.5)

        # Load label schema
        label_schema_path = artifact_path / "label_schema.json"
        if not label_schema_path.exists():
            raise FileNotFoundError(f"Missing label_schema.json in {artifact_dir}")
        with open(label_schema_path) as f:
            self._label_schema = json.load(f)
        self._risk_classes = self._label_schema.get(
            "risk_classes", ["low", "moderate", "high", "critical"]
        )

        # Load TorchScript model
        model_path = artifact_path / "model_scripted.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model_scripted.pt in {artifact_dir}")
        self._model = torch.jit.load(str(model_path), map_location="cpu")
        self._model.eval()

        self._model_version = self._model_config.get("schema_version", "unknown")

        # Warm up with one synthetic forward pass
        self._warmup()

        logger.info(
            "Inference engine loaded: model_version=%s, preprocessing_config_version=%s, "
            "detection_threshold=%.3f",
            self._model_version,
            self._preprocessing_config_version,
            self._detection_threshold,
        )

    def _warmup(self) -> None:
        """Run one forward pass to initialize model internals."""
        dummy_waveform = torch.randn(1, 3, 3000)
        dummy_pgv = torch.randn(1, 1)
        with torch.no_grad():
            self._model(dummy_waveform, dummy_pgv)
        logger.info("Model warmup complete")

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def preprocessing_config_version(self) -> str:
        return self._preprocessing_config_version

    @property
    def risk_classes(self) -> list[str]:
        return self._risk_classes

    @property
    def detection_threshold(self) -> float:
        return self._detection_threshold

    def predict(
        self,
        waveform_np: NDArray[np.floating],
        sampling_rate: int,
        p_arrival_sample: Optional[int] = None,
    ) -> dict[str, Any]:
        """Run full inference pipeline on a preprocessed waveform.

        Args:
            waveform_np: Raw waveform of shape [channels, samples].
            sampling_rate: Sampling rate in Hz.
            p_arrival_sample: P-arrival sample index (optional).

        Returns:
            Structured result dict matching the API schema.
        """
        warnings: list[str] = []

        # Import preprocessing from seismic package — Invariant 1
        from seismic.preprocessing import preprocess_waveform

        # Run preprocessing pipeline
        try:
            processed = preprocess_waveform(
                waveform=waveform_np,
                sampling_rate=sampling_rate,
                config=self._preprocessing_config,
                p_arrival_sample=p_arrival_sample
                if p_arrival_sample is not None
                else self._preprocessing_config["p_arrival_sample_index"],
            )
        except ValueError as e:
            warnings.append(str(e))
            # If sampling rate mismatch, try with default P-arrival
            processed = preprocess_waveform(
                waveform=waveform_np,
                sampling_rate=sampling_rate,
                config=self._preprocessing_config,
                p_arrival_sample=p_arrival_sample
                if p_arrival_sample is not None
                else self._preprocessing_config["p_arrival_sample_index"],
            )

        waveform_tensor = torch.from_numpy(processed["waveform"]).unsqueeze(0)  # [1, 3, 3000]
        pgv_tensor = torch.tensor([[processed["pgv"]]], dtype=torch.float32)  # [1, 1]

        # Forward pass
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self._model(
                waveform_tensor, pgv_tensor,
                active_tasks=["detection", "phase_pick", "magnitude", "risk"],
            )
        inference_ms = int((time.perf_counter() - t0) * 1000)

        # Post-process detection
        det_logit = outputs["detection"].squeeze().item()
        det_prob = float(torch.sigmoid(torch.tensor(det_logit)).item())
        is_seismic = det_prob >= self._detection_threshold

        # Post-process phase picks
        phase_raw = outputs["phase_pick"].squeeze().cpu().numpy()  # [2]
        p_sample_norm = float(np.clip(phase_raw[0], 0.0, 1.0))
        s_sample_norm = float(np.clip(phase_raw[1], 0.0, 1.0))
        window_length = self._preprocessing_config["window_length_samples"]
        p_arrival_sample_out = int(round(p_sample_norm * window_length))
        s_arrival_sample_out = int(round(s_sample_norm * window_length))
        sr = self._preprocessing_config["sampling_rate_hz"]
        p_time_s = p_arrival_sample_out / sr
        s_time_s = s_arrival_sample_out / sr
        sp_interval = s_time_s - p_time_s

        # Post-process magnitude — back-transform from log10(Ml+1)
        mag_log_raw = outputs["magnitude"].squeeze().item()
        ml = float(10**mag_log_raw - 1)

        # Post-process risk
        risk_logits = outputs["risk"].squeeze().cpu().numpy()
        risk_probs = np.exp(risk_logits) / np.exp(risk_logits).sum()  # softmax
        risk_class_idx = int(np.argmax(risk_probs))
        risk_level = self._risk_classes[risk_class_idx]

        return {
            "inference_time_ms": inference_ms,
            "results": {
                "detection": {
                    "is_seismic": is_seismic,
                    "confidence": round(det_prob, 6),
                    "threshold_used": self._detection_threshold,
                },
                "phase_picks": {
                    "p_arrival_sample": p_arrival_sample_out,
                    "s_arrival_sample": s_arrival_sample_out,
                    "p_arrival_time_s": round(p_time_s, 4),
                    "s_arrival_time_s": round(s_time_s, 4),
                    "sp_interval_s": round(sp_interval, 4),
                },
                "magnitude": {
                    "ml": round(ml, 4),
                    "ml_log_raw": round(mag_log_raw, 6),
                },
                "risk": {
                    "level": risk_level,
                    "class_index": risk_class_idx,
                    "probabilities": {
                        cls: round(float(risk_probs[i]), 6)
                        for i, cls in enumerate(self._risk_classes)
                    },
                },
            },
            "warnings": warnings,
            "model_version": self._model_version,
            "preprocessing_config_version": self._preprocessing_config_version,
        }

    def load_from_mseed(
        self,
        file_bytes: bytes,
        stationxml_bytes: Optional[bytes] = None,
    ) -> tuple[NDArray[np.floating], int, Optional[int], list[str]]:
        """Parse MiniSEED file and extract waveform.

        Args:
            file_bytes: Raw bytes of the MiniSEED file.
            stationxml_bytes: Raw bytes of the StationXML file (optional).

        Returns:
            Tuple of (waveform, sampling_rate, p_arrival_sample, warnings).
        """
        warnings: list[str] = []
        p_arrival_sample: Optional[int] = None

        try:
            import io

            from obspy import read as obspy_read
            from obspy import read_inventory

            # Parse MiniSEED
            stream = obspy_read(io.BytesIO(file_bytes))

            # Parse StationXML if provided
            if stationxml_bytes is not None:
                try:
                    inventory = read_inventory(io.BytesIO(stationxml_bytes))
                    stream.attach_response(inventory)
                except Exception as e:
                    warnings.append(f"StationXML parsing failed: {e}")

            # Use first trace (or merge if multiple)
            if len(stream) > 1:
                stream.merge(method=1, fill_value=0)

            trace = stream[0]
            sampling_rate = int(trace.stats.sampling_rate)
            waveform = trace.data.astype(np.float32)

            # Ensure 3-component: pad or slice to 3 channels
            if waveform.ndim == 1:
                waveform = np.stack([waveform, waveform, waveform], axis=0)
            elif waveform.shape[0] > 3:
                waveform = waveform[:3, :]
            elif waveform.shape[0] < 3:
                # Pad with zeros
                pad = np.zeros((3 - waveform.shape[0], waveform.shape[1]), dtype=np.float32)
                waveform = np.concatenate([waveform, pad], axis=0)

        except Exception as e:
            raise ValueError(f"Failed to parse MiniSEED file: {e}") from e

        return waveform, sampling_rate, p_arrival_sample, warnings


def get_engine(artifact_dir: str) -> SeismicInferenceEngine:
    """Get or create the singleton inference engine.

    Thread-safe via threading.Lock.

    Args:
        artifact_dir: Path to the exported model artifact directory.

    Returns:
        The singleton SeismicInferenceEngine instance.
    """
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                _engine = SeismicInferenceEngine(artifact_dir)
    return _engine


def reset_engine() -> None:
    """Reset the singleton (for testing only)."""
    global _engine
    with _lock:
        _engine = None
