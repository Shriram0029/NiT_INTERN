import numpy as np
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
import logging

logger = logging.getLogger(__name__)

class StatisticalAnalyzer:
    def __init__(self):
        self.results = {}
        
    def add_experiment_results(self, baseline_name, metric_values):
        """
        metric_values: list of metric scores (e.g. F1 scores across multiple runs or chunks)
        """
        self.results[baseline_name] = np.array(metric_values)
        
    def compute_summary_stats(self, baseline_name):
        data = self.results.get(baseline_name, [])
        if len(data) == 0:
            return {}
            
        mean = np.mean(data)
        std = np.std(data)
        
        # 95% CI
        n = len(data)
        se = std / np.sqrt(n)
        ci_95 = stats.t.interval(0.95, n-1, loc=mean, scale=se) if n > 1 else (mean, mean)
        
        return {
            "mean": mean,
            "std": std,
            "ci_95_lower": ci_95[0],
            "ci_95_upper": ci_95[1]
        }
        
    def compare_baselines(self, baseline1, baseline2):
        data1 = self.results.get(baseline1, [])
        data2 = self.results.get(baseline2, [])
        
        if len(data1) == 0 or len(data2) == 0 or len(data1) != len(data2):
            logger.warning("Data size mismatch or empty for comparison.")
            return {}
            
        # Wilcoxon signed-rank test
        stat, p_value = stats.wilcoxon(data1, data2, zero_method='zsplit')
        
        # Effect size (Cohen's d)
        mean_diff = np.mean(data1) - np.mean(data2)
        pooled_std = np.sqrt((np.std(data1)**2 + np.std(data2)**2) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0
        
        return {
            "wilcoxon_stat": stat,
            "p_value": p_value,
            "effect_size": cohens_d
        }
        
    def run_full_analysis(self, target_baseline="hybrid"):
        analysis = {}
        
        # 1. Summary Stats
        for name in self.results:
            analysis[name] = {"summary": self.compute_summary_stats(name)}
            
        # 2. Comparisons with Target
        p_values = []
        comparisons = []
        
        for name in self.results:
            if name != target_baseline:
                comp = self.compare_baselines(target_baseline, name)
                if "p_value" in comp:
                    p_values.append(comp["p_value"])
                    comparisons.append(name)
                    analysis[name]["comparison_to_target"] = comp
                    
        # 3. Holm-Bonferroni Correction
        if p_values:
            reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='holm')
            for i, name in enumerate(comparisons):
                analysis[name]["comparison_to_target"]["p_value_holm"] = pvals_corrected[i]
                analysis[name]["comparison_to_target"]["significant"] = reject[i]
                
        return analysis
