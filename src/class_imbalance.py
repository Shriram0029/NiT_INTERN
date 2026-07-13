import numpy as np
from collections import Counter

class ClassImbalanceAnalyzer:
    def __init__(self):
        self.class_counts = Counter()
        self.total_samples = 0
        
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
        
        return {
            "frequencies": freqs,
            "imbalance_factors": imbalance_factors,
            "minority_ratios": minority_ratios,
            "balanced_sampling_ratios": balanced_sampling_ratios
        }
        
    def get_factor(self, label):
        stats = self.compute()
        if "imbalance_factors" in stats:
            return stats["imbalance_factors"].get(label, 1.0)
        return 1.0
