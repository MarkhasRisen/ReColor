"""
Evaluation metrics for ReColor system modules.

This module implements comprehensive evaluation metrics for:
1. Ishihara Module (screening accuracy)
2. Color Correction Module (perceptual accuracy)
3. CNN-Based Color Identifier Module (classification performance)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import warnings


@dataclass
class ClassificationMetrics:
    """Container for classification evaluation results."""
    accuracy: float
    sensitivity: float
    specificity: float
    precision: float
    recall: float
    f1_score: float
    cohen_kappa: float
    confusion_matrix: np.ndarray
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary format."""
        return {
            'accuracy': self.accuracy,
            'sensitivity': self.sensitivity,
            'specificity': self.specificity,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'cohen_kappa': self.cohen_kappa,
            'confusion_matrix': self.confusion_matrix.tolist()
        }


@dataclass
class PerceptualMetrics:
    """Container for perceptual color difference metrics."""
    mean_delta_e: float
    median_delta_e: float
    std_delta_e: float
    percentile_95: float
    delta_l_mean: float
    delta_c_mean: float
    delta_h_mean: float
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary format."""
        return {
            'mean_delta_e': self.mean_delta_e,
            'median_delta_e': self.median_delta_e,
            'std_delta_e': self.std_delta_e,
            'percentile_95': self.percentile_95,
            'delta_l_mean': self.delta_l_mean,
            'delta_c_mean': self.delta_c_mean,
            'delta_h_mean': self.delta_h_mean
        }


class IshiharaEvaluator:
    """
    Evaluation metrics for Ishihara screening module.
    
    Measures diagnostic validity, consistency with clinical standards,
    and effectiveness of CVD detection.
    """
    
    @staticmethod
    def compute_metrics(
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        positive_class: int = 1
    ) -> ClassificationMetrics:
        """
        Compute comprehensive evaluation metrics for Ishihara module.
        
        Args:
            predictions: Predicted classifications (0=normal, 1=CVD)
            ground_truth: True classifications from clinical standard
            positive_class: Label for positive class (CVD detection)
            
        Returns:
            ClassificationMetrics containing all evaluation results
        """
        # Ensure arrays are numpy arrays
        predictions = np.asarray(predictions)
        ground_truth = np.asarray(ground_truth)
        
        # Calculate confusion matrix
        cm = confusion_matrix(ground_truth, predictions)
        
        # Extract TP, TN, FP, FN
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            raise ValueError("Binary classification expected (2x2 confusion matrix)")
        
        # Accuracy: (TP + TN) / (TP + TN + FP + FN)
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        
        # Sensitivity (Recall): TP / (TP + FN)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Specificity: TN / (TN + FP)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Precision (Positive Predictive Value): TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Recall (same as sensitivity for binary classification)
        recall = sensitivity
        
        # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Cohen's Kappa: κ = (p_o - p_e) / (1 - p_e)
        cohen_kappa = cohen_kappa_score(ground_truth, predictions)
        
        return ClassificationMetrics(
            accuracy=accuracy,
            sensitivity=sensitivity,
            specificity=specificity,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            cohen_kappa=cohen_kappa,
            confusion_matrix=cm
        )
    
    @staticmethod
    def interpret_kappa(kappa: float) -> str:
        """
        Interpret Cohen's Kappa value according to Landis & Koch scale.
        
        Args:
            kappa: Cohen's Kappa coefficient
            
        Returns:
            Interpretation string
        """
        if kappa < 0:
            return "Poor agreement (less than chance)"
        elif kappa < 0.20:
            return "Slight agreement"
        elif kappa < 0.40:
            return "Fair agreement"
        elif kappa < 0.60:
            return "Moderate agreement"
        elif kappa < 0.80:
            return "Substantial agreement"
        else:
            return "Almost perfect agreement"


class ColorCorrectionEvaluator:
    """
    Evaluation metrics for Color Correction module (Daltonization + K-Means).
    
    Measures perceptual accuracy using CIEDE2000 color difference formula.
    """
    
    @staticmethod
    def ciede2000(
        lab1: np.ndarray,
        lab2: np.ndarray,
        kL: float = 1.0,
        kC: float = 1.0,
        kH: float = 1.0
    ) -> np.ndarray:
        """
        Calculate CIEDE2000 color difference (ΔE) between two LAB colors.
        
        The CIEDE2000 formula accounts for perceptual non-uniformities in
        color space, providing more accurate measurements of perceived
        color differences than simpler Euclidean distance.
        
        Args:
            lab1: Original LAB color(s) as [..., 3] array
            lab2: Corrected LAB color(s) as [..., 3] array
            kL: Lightness weighting factor (default: 1.0)
            kC: Chroma weighting factor (default: 1.0)
            kH: Hue weighting factor (default: 1.0)
            
        Returns:
            Array of ΔE values representing perceptual color differences
        """
        # Extract L, a, b components
        L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
        L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
        
        # Calculate C (chroma) and h (hue angle)
        C1 = np.sqrt(a1**2 + b1**2)
        C2 = np.sqrt(a2**2 + b2**2)
        
        # Mean chroma
        C_bar = (C1 + C2) / 2.0
        
        # G correction factor for chroma
        G = 0.5 * (1 - np.sqrt(C_bar**7 / (C_bar**7 + 25**7)))
        
        # Adjusted a values
        a1_prime = a1 * (1 + G)
        a2_prime = a2 * (1 + G)
        
        # Recalculate chroma with adjusted a
        C1_prime = np.sqrt(a1_prime**2 + b1**2)
        C2_prime = np.sqrt(a2_prime**2 + b2**2)
        
        # Calculate hue angles
        h1_prime = np.arctan2(b1, a1_prime) % (2 * np.pi)
        h2_prime = np.arctan2(b2, a2_prime) % (2 * np.pi)
        
        # Differences
        delta_L_prime = L2 - L1
        delta_C_prime = C2_prime - C1_prime
        
        # Hue difference (accounting for circular nature)
        delta_h_prime = h2_prime - h1_prime
        delta_h_prime = np.where(
            np.abs(delta_h_prime) > np.pi,
            delta_h_prime - 2 * np.pi * np.sign(delta_h_prime),
            delta_h_prime
        )
        
        # ΔH' (hue difference in chroma units)
        delta_H_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(delta_h_prime / 2)
        
        # Mean values for weighting functions
        L_bar_prime = (L1 + L2) / 2.0
        C_bar_prime = (C1_prime + C2_prime) / 2.0
        h_bar_prime = (h1_prime + h2_prime) / 2.0
        
        # Weighting function SL
        SL = 1 + (0.015 * (L_bar_prime - 50)**2) / np.sqrt(20 + (L_bar_prime - 50)**2)
        
        # Weighting function SC
        SC = 1 + 0.045 * C_bar_prime
        
        # Weighting function SH
        T = (1 - 0.17 * np.cos(h_bar_prime - np.pi/6) +
             0.24 * np.cos(2 * h_bar_prime) +
             0.32 * np.cos(3 * h_bar_prime + np.pi/30) -
             0.20 * np.cos(4 * h_bar_prime - np.pi*63/180))
        SH = 1 + 0.015 * C_bar_prime * T
        
        # Rotation term RT
        delta_theta = np.pi/6 * np.exp(-((h_bar_prime - np.pi*275/180) / (np.pi*25/180))**2)
        RC = 2 * np.sqrt(C_bar_prime**7 / (C_bar_prime**7 + 25**7))
        RT = -RC * np.sin(2 * delta_theta)
        
        # Calculate final ΔE2000
        delta_E = np.sqrt(
            (delta_L_prime / (kL * SL))**2 +
            (delta_C_prime / (kC * SC))**2 +
            (delta_H_prime / (kH * SH))**2 +
            RT * (delta_C_prime / (kC * SC)) * (delta_H_prime / (kH * SH))
        )
        
        return delta_E
    
    @staticmethod
    def compute_metrics(
        original_lab: np.ndarray,
        corrected_lab: np.ndarray
    ) -> PerceptualMetrics:
        """
        Compute perceptual evaluation metrics for color correction.
        
        Args:
            original_lab: Original image in LAB color space
            corrected_lab: Corrected image in LAB color space
            
        Returns:
            PerceptualMetrics containing statistical analysis of color differences
        """
        # Calculate CIEDE2000 for all pixels
        delta_e = ColorCorrectionEvaluator.ciede2000(original_lab, corrected_lab)
        
        # Component differences
        delta_l = corrected_lab[..., 0] - original_lab[..., 0]
        
        # Chroma differences
        c_original = np.sqrt(original_lab[..., 1]**2 + original_lab[..., 2]**2)
        c_corrected = np.sqrt(corrected_lab[..., 1]**2 + corrected_lab[..., 2]**2)
        delta_c = c_corrected - c_original
        
        # Hue differences
        h_original = np.arctan2(original_lab[..., 2], original_lab[..., 1])
        h_corrected = np.arctan2(corrected_lab[..., 2], corrected_lab[..., 1])
        delta_h = h_corrected - h_original
        
        return PerceptualMetrics(
            mean_delta_e=float(np.mean(delta_e)),
            median_delta_e=float(np.median(delta_e)),
            std_delta_e=float(np.std(delta_e)),
            percentile_95=float(np.percentile(delta_e, 95)),
            delta_l_mean=float(np.mean(np.abs(delta_l))),
            delta_c_mean=float(np.mean(np.abs(delta_c))),
            delta_h_mean=float(np.mean(np.abs(delta_h)))
        )
    
    @staticmethod
    def interpret_delta_e(delta_e: float) -> str:
        """
        Interpret CIEDE2000 ΔE value.
        
        Args:
            delta_e: CIEDE2000 color difference value
            
        Returns:
            Interpretation string
        """
        if delta_e < 1.0:
            return "Imperceptible difference"
        elif delta_e < 2.0:
            return "Perceptible through close observation"
        elif delta_e < 3.5:
            return "Perceptible at a glance"
        elif delta_e < 5.0:
            return "Clear difference"
        else:
            return "Colors appear to be different"


class CNNColorEvaluator:
    """
    Evaluation metrics for CNN-Based Color Identifier module.
    
    Measures classification accuracy, robustness, and applicability
    for assisting anomalous trichromats.
    """
    
    @staticmethod
    def compute_multiclass_metrics(
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Compute comprehensive multiclass classification metrics.
        
        Args:
            predictions: Predicted class labels
            ground_truth: True class labels
            class_names: Optional list of class names for labeling
            
        Returns:
            Dictionary containing per-class and overall metrics
        """
        # Ensure arrays are numpy arrays
        predictions = np.asarray(predictions)
        ground_truth = np.asarray(ground_truth)
        
        # Compute confusion matrix
        cm = confusion_matrix(ground_truth, predictions)
        n_classes = cm.shape[0]
        
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(n_classes)]
        
        # Per-class metrics
        per_class_metrics = {}
        
        for i in range(n_classes):
            # Extract TP, FP, FN, TN for each class
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
            # Precision: TP / (TP + FP)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            
            # Recall: TP / (TP + FN)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            per_class_metrics[class_names[i]] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': int(cm[i, :].sum())
            }
        
        # Overall metrics
        total_samples = cm.sum()
        correct_predictions = np.trace(cm)
        overall_accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        # Macro-averaged metrics (unweighted mean of per-class metrics)
        macro_precision = np.mean([m['precision'] for m in per_class_metrics.values()])
        macro_recall = np.mean([m['recall'] for m in per_class_metrics.values()])
        macro_f1 = np.mean([m['f1_score'] for m in per_class_metrics.values()])
        
        # Weighted metrics (weighted by support)
        total_support = sum([m['support'] for m in per_class_metrics.values()])
        weighted_precision = sum([m['precision'] * m['support'] for m in per_class_metrics.values()]) / total_support if total_support > 0 else 0.0
        weighted_recall = sum([m['recall'] * m['support'] for m in per_class_metrics.values()]) / total_support if total_support > 0 else 0.0
        weighted_f1 = sum([m['f1_score'] * m['support'] for m in per_class_metrics.values()]) / total_support if total_support > 0 else 0.0
        
        return {
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': per_class_metrics,
            'overall_accuracy': overall_accuracy,
            'macro_avg': {
                'precision': macro_precision,
                'recall': macro_recall,
                'f1_score': macro_f1
            },
            'weighted_avg': {
                'precision': weighted_precision,
                'recall': weighted_recall,
                'f1_score': weighted_f1
            }
        }
    
    @staticmethod
    def plot_confusion_matrix(
        cm: np.ndarray,
        class_names: List[str],
        normalize: bool = False
    ) -> Dict:
        """
        Generate confusion matrix visualization data.
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
            normalize: Whether to normalize by true labels
            
        Returns:
            Dictionary with matrix data and labels
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        return {
            'matrix': cm.tolist(),
            'class_names': class_names,
            'normalized': normalize
        }
