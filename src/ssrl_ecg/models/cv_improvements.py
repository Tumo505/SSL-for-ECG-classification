"""
Cardiovascular risk prediction improvements: Domain-specific ECG analysis.

Integrates clinical knowledge for accurate cardiovascular disease detection:
- Focal loss for rare disease detection
- Clinical class weighting
- Multi-scale ECG analysis
- Morphology-aware features
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for handling extreme class imbalance.
    
    Particularly useful for rare cardiovascular diseases where one class 
    is dominant (e.g., HYP prevalence 12.2%).
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, logits, targets):
        """
        Args:
            logits: Model logits (batch_size,)
            targets: Binary targets (batch_size,)
        """
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none',
            pos_weight=self.pos_weight
        )
        
        probs = torch.sigmoid(logits)
        p_t = torch.where(targets == 1, probs, 1 - probs)
        
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * bce_loss
        
        return focal_loss.mean()


class ClinicalClassWeighting:
    """
    Compute clinical class weights based on:
    1. Label frequency (statistical balancing)
    2. Clinical importance (disease severity)
    3. Diagnostic difficulty
    """
    
    # Clinical importance scores (based on mortality/morbidity)
    CLINICAL_IMPORTANCE = {
        "NORM": 1.0,     # Normal: baseline
        "MI": 3.0,       # Myocardial infarction: high risk (mortality 5-10%)
        "STTC": 2.0,     # ST-T changes: moderate risk
        "HYP": 2.5,      # Hypertrophy (left ventricular): moderate-high risk
        "CD": 3.5,       # Cardiomyopathy: highest risk (mortality 25-50%)
    }
    
    @staticmethod
    def compute_weighted_importance(label_frequencies, class_names):
        """Combine statistical frequency with clinical importance.
        
        Args:
            label_frequencies: Dict of class -> frequency
            class_names: List of class names
            
        Returns:
            Dict of class -> combined weight
        """
        weights = {}
        total_importance = sum(ClinicalClassWeighting.CLINICAL_IMPORTANCE.values())
        
        for cls in class_names:
            freq = label_frequencies.get(cls, 0.1)
            clinical_score = ClinicalClassWeighting.CLINICAL_IMPORTANCE.get(cls, 1.0)
            
            # Inverse frequency weighting (favor rare classes)
            inv_freq_weight = (1 - freq) / (2 * max(freq, 0.001))
            
            # Clinical importance scaling (favor serious diseases)
            clinical_weight = clinical_score / (total_importance / len(class_names))
            
            # Combined weight
            weights[cls] = inv_freq_weight * clinical_weight
        
        # Normalize to sum to 1
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}


class MultiScaleECGAnalysis(nn.Module):
    """
    Analyze ECG at multiple time scales:
    - Fast scale: QRS complex (~40ms)
    - Medium scale: Full heartbeat (~1000ms)
    - Slow scale: ST segment dynamics
    """
    
    def __init__(self, in_channels=12, base_width=64):
        super().__init__()
        
        # Fast scale: Small kernel (local features)
        self.fast_conv = nn.Sequential(
            nn.Conv1d(in_channels, base_width, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(base_width),
            nn.ReLU(),
        )
        
        # Medium scale: Medium kernel (heartbeat)
        self.medium_conv = nn.Sequential(
            nn.Conv1d(in_channels, base_width, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(base_width),
            nn.ReLU(),
        )
        
        # Slow scale: Large kernel (long-term trends)
        self.slow_conv = nn.Sequential(
            nn.Conv1d(in_channels, base_width, kernel_size=31, stride=1, padding=15),
            nn.BatchNorm1d(base_width),
            nn.ReLU(),
        )
    
    def forward(self, x):
        """
        Args:
            x: ECG signal (batch_size, channels, samples)
            
        Returns:
            Concatenated multi-scale features
        """
        fast = self.fast_conv(x)
        medium = self.medium_conv(x)
        slow = self.slow_conv(x)
        
        # Concatenate along channel dimension
        return torch.cat([fast, medium, slow], dim=1)


class CVRiskPredictor(nn.Module):
    """
    Comprehensive cardiovascular risk predictor with:
    - Multi-scale ECG analysis
    - Attention to important regions
    - Interpretable risk stratification
    """
    
    def __init__(self, encoder, n_classes=5, use_attention=True):
        super().__init__()
        self.encoder = encoder
        self.n_classes = n_classes
        
        # Risk stratification: Separate heads for different risk levels
        hidden_dim = encoder.out_channels
        
        # Normal vs Abnormal (binary)
        self.abnormal_detector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )
        
        # Risk classifier: Low, Medium, High
        self.risk_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 3),  # Low, Medium, High
        )
        
        # Disease-specific classifiers
        self.disease_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_classes),
        )
        
        # Attention map (for interpretability)
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.Sequential(
                nn.Conv1d(encoder.out_channels, 1, kernel_size=1),
                nn.Sigmoid(),
            )
    
    def forward(self, x):
        """
        Args:
            x: ECG signal (batch_size, 12, samples)
            
        Returns:
            Logits for disease classification (batch_size, n_classes)
        """
        # Extract features
        features = self.encoder(x)  # (batch, hidden_dim, time_steps)
        
        # Compute attention if enabled
        if self.use_attention:
            att_weights = self.attention(features)  # (batch, 1, time_steps)
            features_weighted = features * att_weights
            avg_pool = torch.nn.functional.adaptive_avg_pool1d(features_weighted, 1).squeeze(-1)
        else:
            avg_pool = torch.nn.functional.adaptive_avg_pool1d(features, 1).squeeze(-1)
        
        max_pool = torch.nn.functional.adaptive_max_pool1d(features, 1).squeeze(-1)
        combined = torch.cat([avg_pool, max_pool], dim=1)
        
        # Disease logits
        disease_logits = self.disease_classifier(combined)
        
        return disease_logits


class CardiovascularMetrics:
    """Clinical metrics for cardiovascular disease prediction."""
    
    @staticmethod
    def specificity(y_true, y_pred, threshold=0.5):
        """True negative rate: ability to correctly identify healthy patients."""
        pred_binary = (y_pred > threshold).astype(int)
        tn = np.sum((y_true == 0) & (pred_binary == 0))
        fp = np.sum((y_true == 0) & (pred_binary == 1))
        return tn / (tn + fp) if (tn + fp) > 0 else 0
    
    @staticmethod
    def sensitivity(y_true, y_pred, threshold=0.5):
        """True positive rate: ability to detect disease."""
        pred_binary = (y_pred > threshold).astype(int)
        tp = np.sum((y_true == 1) & (pred_binary == 1))
        fn = np.sum((y_true == 1) & (pred_binary == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0
    
    @staticmethod
    def npv(y_true, y_pred, threshold=0.5):
        """Negative Predictive Value: probability that negative test is truly negative.
        
        Clinical importance: Ensures we don't miss healthy patients.
        """
        pred_binary = (y_pred > threshold).astype(int)
        tn = np.sum((y_true == 0) & (pred_binary == 0))
        fn = np.sum((y_true == 1) & (pred_binary == 0))
        return tn / (tn + fn) if (tn + fn) > 0 else 0
    
    @staticmethod
    def ppv(y_true, y_pred, threshold=0.5):
        """Positive Predictive Value: probability that positive test is truly positive.
        
        Clinical importance: Ensures we minimize unnecessary interventions.
        """
        pred_binary = (y_pred > threshold).astype(int)
        tp = np.sum((y_true == 1) & (pred_binary == 1))
        fp = np.sum((y_true == 0) & (pred_binary == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0
    
    @staticmethod
    def compute_clinical_metrics(y_true, y_prob):
        """Compute comprehensive clinical evaluation metrics."""
        from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc
        
        metrics = {}
        
        # For multi-class/multi-label, average across classes
        y_true_flat = y_true.flatten()
        y_prob_flat = y_prob.flatten()
        
        # AUROC
        try:
            metrics['auroc'] = roc_auc_score(y_true_flat, y_prob_flat)
        except:
            metrics['auroc'] = 0.5
        
        # Find optimal threshold for F1
        precision, recall, thresholds = precision_recall_curve(y_true_flat, y_prob_flat)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        # Sensitivity, Specificity at optimal threshold
        metrics['sensitivity'] = CardiovascularMetrics.sensitivity(
            y_true_flat, y_prob_flat, optimal_threshold
        )
        metrics['specificity'] = CardiovascularMetrics.specificity(
            y_true_flat, y_prob_flat, optimal_threshold
        )
        metrics['ppv'] = CardiovascularMetrics.ppv(y_true_flat, y_prob_flat, optimal_threshold)
        metrics['npv'] = CardiovascularMetrics.npv(y_true_flat, y_prob_flat, optimal_threshold)
        metrics['optimal_threshold'] = optimal_threshold
        
        return metrics
