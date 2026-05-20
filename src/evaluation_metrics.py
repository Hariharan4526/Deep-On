"""ENHANCEMENT-05: Evaluation Metrics Module.

Comprehensive metrics for model evaluation:
  - ROC-AUC score
  - F1-score
  - Confusion matrix
  - False positive/negative analysis
  - Precision, Recall, Specificity

Complements existing utils.compute_metrics.
"""
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    f1_score, confusion_matrix,
    precision_score, recall_score,
    precision_recall_curve
)
from .utils import setup_logger

logger = setup_logger(__name__)


class EvaluationMetrics:
    """Comprehensive evaluation metrics for deepfake detection."""

    def __init__(self):
        logger.info("EvaluationMetrics ready")

    # ── Basic metrics ─────────────────────────────────────────────────────────

    @staticmethod
    def compute_roc_auc(y_true: List[int], y_probs: List[float]) -> float:
        """Compute ROC-AUC score.
        
        Args:
            y_true: Ground truth binary labels (0/1).
            y_probs: Predicted probabilities (0-1).
            
        Returns:
            ROC-AUC score [0, 1].
        """
        try:
            return float(roc_auc_score(y_true, y_probs))
        except Exception as e:
            logger.warning(f"ROC-AUC computation failed: {e}")
            return 0.5

    @staticmethod
    def compute_f1(y_true: List[int], y_pred: List[int]) -> float:
        """Compute F1-score.
        
        Returns:
            F1-score [0, 1].
        """
        try:
            return float(f1_score(y_true, y_pred, zero_division=0))
        except Exception as e:
            logger.warning(f"F1 computation failed: {e}")
            return 0.0

    @staticmethod
    def compute_confusion_matrix(
        y_true: List[int], y_pred: List[int]
    ) -> Dict[str, int]:
        """Compute confusion matrix and metrics.
        
        Returns dict with:
            - tn, fp, fn, tp (confusion matrix entries)
            - accuracy, precision, recall, specificity, f1
        """
        try:
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            
            n = len(y_true)
            accuracy = (tp + tn) / n if n > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            return {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "specificity": float(specificity),
                "f1": float(f1),
            }
        except Exception as e:
            logger.warning(f"Confusion matrix computation failed: {e}")
            return {}

    # ── Detailed analysis ─────────────────────────────────────────────────────

    @staticmethod
    def analyze_false_positives(
        y_true: List[int], y_pred: List[int], y_probs: List[float]
    ) -> Dict[str, any]:
        """Analyze false positive detections.
        
        Returns dict with FP statistics and analysis.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_probs = np.array(y_probs)
        
        # False positives: predicted fake (1) but actually real (0)
        fp_mask = (y_pred == 1) & (y_true == 0)
        fp_indices = np.where(fp_mask)[0]
        
        # False negatives: predicted real (0) but actually fake (1)
        fn_mask = (y_pred == 0) & (y_true == 1)
        fn_indices = np.where(fn_mask)[0]
        
        analysis = {
            "false_positive_count": int(np.sum(fp_mask)),
            "false_negative_count": int(np.sum(fn_mask)),
            "total_false": int(np.sum(fp_mask) + np.sum(fn_mask)),
        }
        
        if len(fp_indices) > 0:
            fp_probs = y_probs[fp_indices]
            analysis["fp_prob_mean"] = float(fp_probs.mean())
            analysis["fp_prob_std"] = float(fp_probs.std())
            analysis["fp_prob_min"] = float(fp_probs.min())
            analysis["fp_prob_max"] = float(fp_probs.max())
        
        if len(fn_indices) > 0:
            fn_probs = y_probs[fn_indices]
            analysis["fn_prob_mean"] = float(fn_probs.mean())
            analysis["fn_prob_std"] = float(fn_probs.std())
            analysis["fn_prob_min"] = float(fn_probs.min())
            analysis["fn_prob_max"] = float(fn_probs.max())
        
        return analysis

    @staticmethod
    def compute_roc_curve(
        y_true: List[int], y_probs: List[float]
    ) -> Dict[str, any]:
        """Compute ROC curve points.
        
        Returns dict with fpr, tpr, thresholds, auc.
        """
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_probs)
            auc_score = auc(fpr, tpr)
            
            return {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist(),
                "auc": float(auc_score),
            }
        except Exception as e:
            logger.warning(f"ROC curve computation failed: {e}")
            return {}

    @staticmethod
    def compute_precision_recall_curve(
        y_true: List[int], y_probs: List[float]
    ) -> Dict[str, any]:
        """Compute precision-recall curve points.
        
        Returns dict with precision, recall, thresholds, auc.
        """
        try:
            precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
            pr_auc = auc(recall, precision)
            
            return {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "thresholds": thresholds.tolist(),
                "auc": float(pr_auc),
            }
        except Exception as e:
            logger.warning(f"PR curve computation failed: {e}")
            return {}

    # ── Threshold analysis ────────────────────────────────────────────────────

    @staticmethod
    def find_optimal_threshold(
        y_true: List[int], y_probs: List[float], metric: str = "f1"
    ) -> Dict[str, float]:
        """Find optimal classification threshold.
        
        Args:
            y_true: Ground truth labels.
            y_probs: Predicted probabilities.
            metric: One of 'f1', 'youden', 'precision_recall'
            
        Returns dict with optimal threshold and metrics at that point.
        """
        y_true = np.array(y_true)
        y_probs = np.array(y_probs)
        
        thresholds = np.linspace(0, 1, 100)
        scores = []
        
        for th in thresholds:
            y_pred = (y_probs >= th).astype(int)
            
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            
            if metric == "f1":
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            elif metric == "youden":
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                score = tpr - fpr
            else:  # precision_recall
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                score = (precision + recall) / 2
            
            scores.append(score)
        
        optimal_idx = np.argmax(scores)
        optimal_threshold = float(thresholds[optimal_idx])
        optimal_score = float(scores[optimal_idx])
        
        return {
            "optimal_threshold": optimal_threshold,
            "optimal_score": optimal_score,
            "metric": metric,
        }

    # ── Per-class analysis ────────────────────────────────────────────────────

    @staticmethod
    def per_class_metrics(
        y_true: List[int], y_pred: List[int]
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics per class.
        
        Returns dict with metrics for each class (0 and 1).
        """
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        
        if cm.shape != (2, 2):
            logger.warning("Confusion matrix not 2x2")
            return {}
        
        tn, fp, fn, tp = cm.ravel()
        
        metrics_0 = {
            "precision": float(tn / (tn + fn)) if (tn + fn) > 0 else 0,
            "recall": float(tn / (tn + fp)) if (tn + fp) > 0 else 0,
            "f1": 0.0,
        }
        metrics_0["f1"] = 2 * (metrics_0["precision"] * metrics_0["recall"]) / \
                          (metrics_0["precision"] + metrics_0["recall"]) \
                          if (metrics_0["precision"] + metrics_0["recall"]) > 0 else 0
        
        metrics_1 = {
            "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0,
            "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
            "f1": 0.0,
        }
        metrics_1["f1"] = 2 * (metrics_1["precision"] * metrics_1["recall"]) / \
                          (metrics_1["precision"] + metrics_1["recall"]) \
                          if (metrics_1["precision"] + metrics_1["recall"]) > 0 else 0
        
        return {
            "class_0_authentic": metrics_0,
            "class_1_deepfake": metrics_1,
        }

    # ── Comprehensive evaluation ──────────────────────────────────────────────

    def evaluate_comprehensive(
        self, y_true: List[int], y_pred: List[int], y_probs: List[float]
    ) -> Dict[str, any]:
        """Run comprehensive evaluation.
        
        Returns dict with all metrics.
        """
        roc_auc = self.compute_roc_auc(y_true, y_probs)
        f1 = self.compute_f1(y_true, y_pred)
        cm = self.compute_confusion_matrix(y_true, y_pred)
        fp_analysis = self.analyze_false_positives(y_true, y_pred, y_probs)
        roc_curve = self.compute_roc_curve(y_true, y_probs)
        pr_curve = self.compute_precision_recall_curve(y_true, y_probs)
        optimal_th = self.find_optimal_threshold(y_true, y_probs, metric="f1")
        per_class = self.per_class_metrics(y_true, y_pred)
        
        return {
            "roc_auc": roc_auc,
            "f1": f1,
            "confusion_matrix": cm,
            "false_positive_analysis": fp_analysis,
            "roc_curve": roc_curve,
            "precision_recall_curve": pr_curve,
            "optimal_threshold": optimal_th,
            "per_class_metrics": per_class,
        }
