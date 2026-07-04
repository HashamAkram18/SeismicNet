"""Evaluation metrics for all four seismic tasks.

Computes per-task metrics from model outputs and ground truth labels.
All metrics except risk_confusion_matrix are scalars.
The confusion matrix is a 4×4 list[list[int]] logged as JSON artifact.

Defined in docs/03_model_architecture.md § metrics.py.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor


class SeismicMetrics:
    """Computes and aggregates evaluation metrics for all tasks.

    Usage:
        metrics = SeismicMetrics()
        for batch in dataloader:
            outputs = model(batch["waveform"], batch["pgv"])
            metrics.update(outputs, batch)
        results = metrics.compute()

    Metric specifications (docs/03_model_architecture.md § metrics.py):
        Task A: detection_precision, detection_recall, detection_f1,
                detection_roc_auc, detection_false_alarm_rate
        Task B: p_pick_mae_samples, s_pick_mae_samples,
                p_pick_within_0.1s, s_pick_within_0.1s
        Task C: magnitude_rmse, magnitude_mae, magnitude_bias
        Task D: risk_accuracy, risk_f1_macro, risk_confusion_matrix,
                risk_critical_recall
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all metric accumulators."""
        # Task A accumulators
        self._det_preds: list[Tensor] = []
        self._det_labels: list[Tensor] = []

        # Task B accumulators
        self._p_pick_errors: list[Tensor] = []
        self._s_pick_errors: list[Tensor] = []
        self._n_seismic = 0

        # Task C accumulators
        self._mag_preds: list[Tensor] = []
        self._mag_labels: list[Tensor] = []

        # Task D accumulators
        self._risk_preds: list[Tensor] = []
        self._risk_labels: list[Tensor] = []

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
        is_noise = batch["is_noise"].squeeze(-1)  # [B]
        seismic_mask = ~is_noise  # [B]

        # Task A: Detection — accumulate all windows
        if "detection" in outputs:
            self._det_preds.append(outputs["detection"].squeeze(-1).detach().cpu())
            self._det_labels.append(batch["label_detection"].squeeze(-1).detach().cpu())

        # Task B: Phase pick — seismic only
        if "phase_pick" in outputs and seismic_mask.any():
            p_pred = outputs["phase_pick"][seismic_mask, 0]  # [n_seismic]
            s_pred = outputs["phase_pick"][seismic_mask, 1]
            p_target = batch["label_phase_pick"][seismic_mask, 0]
            s_target = batch["label_phase_pick"][seismic_mask, 1]

            # Convert normalized [0,1] to sample indices (×3000)
            p_errors = torch.abs(p_pred - p_target) * 3000  # in samples
            s_errors = torch.abs(s_pred - s_target) * 3000

            self._p_pick_errors.append(p_errors.detach().cpu())
            self._s_pick_errors.append(s_errors.detach().cpu())

        # Task C: Magnitude — seismic only
        if "magnitude" in outputs and seismic_mask.any():
            mag_pred = outputs["magnitude"].squeeze(-1)[seismic_mask]
            mag_target = batch["label_magnitude"].squeeze(-1)[seismic_mask]
            self._mag_preds.append(mag_pred.detach().cpu())
            self._mag_labels.append(mag_target.detach().cpu())

        # Task D: Risk — seismic only
        if "risk" in outputs and seismic_mask.any():
            risk_pred = outputs["risk"][seismic_mask].argmax(dim=-1)
            risk_target = batch["label_risk"].squeeze(-1)[seismic_mask]
            self._risk_preds.append(risk_pred.detach().cpu())
            self._risk_labels.append(risk_target.detach().cpu())

    def compute(self) -> dict[str, Any]:
        """Compute all accumulated metrics.

        Returns:
            Dict with all metric keys as specified in docs/03_model_architecture.md.
            risk_confusion_matrix is list[list[int]] (4×4).
        """
        results: dict[str, Any] = {}

        # --- Task A: Detection ---
        if self._det_preds:
            preds = torch.cat(self._det_preds)
            labels = torch.cat(self._det_labels)
            probs = torch.sigmoid(preds)

            # Binarize at 0.5 threshold
            binary = (probs >= 0.5).long()
            labels_long = labels.long()

            tp = ((binary == 1) & (labels_long == 1)).sum().item()
            fp = ((binary == 1) & (labels_long == 0)).sum().item()
            fn = ((binary == 0) & (labels_long == 1)).sum().item()
            tn = ((binary == 0) & (labels_long == 0)).sum().item()

            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            false_alarm_rate = fp / max(fp + tn, 1)

            # ROC AUC via trapezoidal rule
            sorted_indices = torch.argsort(probs, descending=True)
            sorted_labels = labels_long[sorted_indices]
            tps = sorted_labels.cumsum(dim=0).float()
            fps = (1 - sorted_labels).cumsum(dim=0).float()
            tpr = tps / max(tp + fn, 1)
            fpr = fps / max(fp + tn, 1)
            roc_auc = torch.trapezoid(tpr, fpr).item() if len(tpr) > 1 else 0.5

            results["detection_precision"] = precision
            results["detection_recall"] = recall
            results["detection_f1"] = f1
            results["detection_roc_auc"] = roc_auc
            results["detection_false_alarm_rate"] = false_alarm_rate
        else:
            results["detection_precision"] = 0.0
            results["detection_recall"] = 0.0
            results["detection_f1"] = 0.0
            results["detection_roc_auc"] = 0.5
            results["detection_false_alarm_rate"] = 0.0

        # --- Task B: Phase pick ---
        if self._p_pick_errors:
            p_err = torch.cat(self._p_pick_errors)
            s_err = torch.cat(self._s_pick_errors)
            results["p_pick_mae_samples"] = p_err.mean().item()
            results["s_pick_mae_samples"] = s_err.mean().item()
            results["p_pick_within_0.1s"] = (p_err < 10).float().mean().item()
            results["s_pick_within_0.1s"] = (s_err < 10).float().mean().item()
        else:
            results["p_pick_mae_samples"] = 0.0
            results["s_pick_mae_samples"] = 0.0
            results["p_pick_within_0.1s"] = 0.0
            results["s_pick_within_0.1s"] = 0.0

        # --- Task C: Magnitude ---
        if self._mag_preds:
            pred = torch.cat(self._mag_preds)
            target = torch.cat(self._mag_labels)

            # Back-transform from log10(Ml+1) to Ml scale
            pred_ml = 10 ** pred - 1
            target_ml = 10 ** target - 1

            results["magnitude_rmse"] = torch.sqrt(torch.mean((pred_ml - target_ml) ** 2)).item()
            results["magnitude_mae"] = torch.mean(torch.abs(pred_ml - target_ml)).item()
            results["magnitude_bias"] = torch.mean(pred_ml - target_ml).item()
        else:
            results["magnitude_rmse"] = 0.0
            results["magnitude_mae"] = 0.0
            results["magnitude_bias"] = 0.0

        # --- Task D: Risk ---
        if self._risk_preds:
            preds = torch.cat(self._risk_preds)
            labels = torch.cat(self._risk_labels)

            results["risk_accuracy"] = (preds == labels).float().mean().item()

            # Macro F1
            f1_per_class = []
            for c in range(4):
                tp_c = ((preds == c) & (labels == c)).sum().item()
                fp_c = ((preds == c) & (labels != c)).sum().item()
                fn_c = ((preds != c) & (labels == c)).sum().item()
                p_c = tp_c / max(tp_c + fp_c, 1)
                r_c = tp_c / max(tp_c + fn_c, 1)
                f1_c = 2 * p_c * r_c / max(p_c + r_c, 1e-8)
                f1_per_class.append(f1_c)
            results["risk_f1_macro"] = float(np.mean(f1_per_class))

            # Confusion matrix
            cm = [[0] * 4 for _ in range(4)]
            for p, t in zip(preds.tolist(), labels.tolist()):
                cm[int(t)][int(p)] += 1
            results["risk_confusion_matrix"] = cm

            # Critical recall (class 3)
            critical_mask = labels == 3
            if critical_mask.sum() > 0:
                results["risk_critical_recall"] = (preds[critical_mask] == 3).float().mean().item()
            else:
                results["risk_critical_recall"] = 0.0
        else:
            results["risk_accuracy"] = 0.0
            results["risk_f1_macro"] = 0.0
            results["risk_confusion_matrix"] = [[0] * 4 for _ in range(4)]
            results["risk_critical_recall"] = 0.0

        return results
