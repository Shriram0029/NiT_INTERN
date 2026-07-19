import numpy as np
from collections import Counter
import torch

class ClassImbalanceAnalyzer:
    def __init__(self, num_classes):
        self.class_counts = Counter()
        self.total_samples = 0
        self.num_classes = num_classes
        
    def fit(self, labels):
        self.class_counts.update(labels)
        self.total_samples += len(labels)
        
    def compute(self):
        if not self.class_counts:
            return {}
            
        freqs = {k: v / self.total_samples for k, v in self.class_counts.items()}
        max_freq = max(freqs.values())
        
        imbalance_factors = {k: max_freq / f if f > 0 else 1.0 for k, f in freqs.items()}
        minority_ratios = {k: f / max_freq if max_freq > 0 else 1.0 for k, f in freqs.items()}
        balanced_sampling_ratios = {k: (1.0 / len(freqs)) / f if f > 0 else 1.0 for k, f in freqs.items()}
        inverse_frequencies = {k: 1.0 / f if f > 0 else 0.0 for k, f in freqs.items()}
        
        # Calculate sample weights for WeightedRandomSampler
        weights = []
        for i in range(self.num_classes):
            w = inverse_frequencies.get(i, 0.0)
            weights.append(w)
            
        return {
            "frequencies": freqs,
            "imbalance_factors": imbalance_factors,
            "minority_ratios": minority_ratios,
            "balanced_sampling_ratios": balanced_sampling_ratios,
            "inverse_frequencies": inverse_frequencies,
            "class_weights": weights
        }
        
    def get_factor(self, label):
        stats = self.compute()
        if "imbalance_factors" in stats:
            return stats["imbalance_factors"].get(label, 1.0)
        return 1.0

    def get_class_weights_tensor(self, device='cpu'):
        stats = self.compute()
        weights = stats.get("class_weights", [1.0] * self.num_classes)
        # Normalize weights so they don't blow up gradients
        if sum(weights) > 0:
            weights = [w / sum(weights) * self.num_classes for w in weights]
        return torch.tensor(weights, dtype=torch.float32).to(device)
