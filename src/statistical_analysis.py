import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import pandas as pd
import os

class StatisticalAnalyzer:
    def __init__(self, results_dir="results"):
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        
    def _compute_effect_size(self, x, y):
        """Computes Cohen's d effect size"""
        nx, ny = len(x), len(y)
        if nx < 2 or ny < 2: return 0.0
        
        dof = nx + ny - 2
        var_x = np.var(x, ddof=1)
        var_y = np.var(y, ddof=1)
        pooled_std = np.sqrt(((nx - 1) * var_x + (ny - 1) * var_y) / dof)
        
        if pooled_std == 0: return 0.0
        return (np.mean(x) - np.mean(y)) / pooled_std

    def run_analysis(self, baseline_metrics_dict, target_metric="macro_f1"):
        """
        baseline_metrics_dict: dict { "baseline_name": [list of metric values across chunks] }
        """
        results = []
        baselines = list(baseline_metrics_dict.keys())
        
        # We need "Hybrid GA + GWO" as the proposed method to compare against
        proposed_name = "Hybrid GA + GWO"
        if proposed_name not in baselines:
            # Try to find something with hybrid in it
            for b in baselines:
                if "hybrid" in b.lower():
                    proposed_name = b
                    break
                    
        proposed_data = baseline_metrics_dict.get(proposed_name, [])
        
        for baseline in baselines:
            data = baseline_metrics_dict[baseline]
            if len(data) == 0:
                continue
                
            mean = np.mean(data)
            std = np.std(data)
            
            # 95% CI
            n = len(data)
            sem = stats.sem(data) if n > 1 else 0
            ci = stats.t.interval(0.95, n-1, loc=mean, scale=sem) if n > 1 and sem > 0 else (mean, mean)
            
            # Compare to proposed if not self
            wilcoxon_p = 1.0
            effect_size = 0.0
            
            if baseline != proposed_name and len(data) == len(proposed_data) and len(data) >= 2:
                try:
                    stat, wilcoxon_p = stats.wilcoxon(proposed_data, data)
                except ValueError:
                    wilcoxon_p = 1.0
                
                effect_size = self._compute_effect_size(proposed_data, data)
                
            results.append({
                "Method": baseline,
                "Mean": mean,
                "Std": std,
                "CI_Lower": ci[0],
                "CI_Upper": ci[1],
                "Wilcoxon_p": wilcoxon_p,
                "Effect_Size": effect_size
            })
            
        if len(results) == 0:
            return pd.DataFrame()
            
        df = pd.DataFrame(results)
        
        # Holm Correction on p-values
        # Only apply to those that were compared
        p_vals = df['Wilcoxon_p'].values
        valid_idx = np.where(p_vals < 1.0)[0]
        
        if len(valid_idx) > 0:
            reject, pvals_corrected, _, _ = multipletests(p_vals[valid_idx], alpha=0.05, method='holm')
            df.loc[valid_idx, 'Wilcoxon_p_Holm'] = pvals_corrected
        else:
            df['Wilcoxon_p_Holm'] = p_vals
            
        df.to_csv(os.path.join(self.results_dir, "statistical_analysis.csv"), index=False)
        return df
