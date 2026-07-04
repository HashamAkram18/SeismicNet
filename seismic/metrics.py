"""Evaluation metrics for all four seismic tasks.

Computes per-task metrics from model outputs and ground truth labels.
Defined in docs/03_model_architecture.md § metrics.py.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


class SeismicMetrics:
    """Computes and aggregates evaluation metrics for all tasks.

    Metric specifications (see docs/03_model_architecture.md § metrics.py):
        Task A: precision, recall, f1, roc_auc, false_alarm_rate
        Task B: p/s_pick_mae_samples, p/s_pick_within_0.1s
        Task C: magnitude_rmse, magnitude_mae, magnitude_bias
        Task D: risk_accuracy, risk_f1_macro, risk_confusion_matrix, risk_critical_recall
    """

    def __init__(self) -> None:
        # TODO: initialize metric accumulators
        pass

    def reset(self) -> None:
        """Reset all metric accumulators."""
        # TODO: implement

    def update(
        self,
        outputs: dict[str, Tensor],
        batch: dict[str, Tensor],
    ) -> None:
        """Accumulate metrics for a batch.

        Args:
            outputs: Model output dict.
            batch: Batch dict with label tensors.
        """
        # TODO: implement

    def compute(self) -> dict[str, float]:
        """Compute all accumulated metrics.

        Returns:
            Dict with all metric keys as specified in docs/03_model_architecture.md.
        """
        # TODO: implement
        raise NotImplementedError
