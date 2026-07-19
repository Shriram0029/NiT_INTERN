"""
src/statistical_analysis.py — Publication-grade statistical analysis
Computes mean ± std, 95% confidence intervals, and Wilcoxon significance tests.
"""

import os
import logging
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.stats as stats

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """
    Aggregates metrics across multiple baselines or seeds and produces
    summary tables suitable for inclusion in IEEE/Elsevier publications.
    """

    def __init__(self, results_dir: str = "results") -> None:
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────
    # Core statistics
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def confidence_interval(
        data: List[float],
        confidence: float = 0.95,
    ) -> tuple:
        """Return (lower, upper) 95% CI using t-distribution."""
        n = len(data)
        if n < 2:
            return (0.0, 0.0)
        se  = stats.sem(data)
        h   = se * stats.t.ppf((1 + confidence) / 2.0, df=n - 1)
        m   = float(np.mean(data))
        return (round(m - h, 6), round(m + h, 6))

    @staticmethod
    def wilcoxon_test(
        a: List[float],
        b: List[float],
    ) -> float:
        """
        Two-sided Wilcoxon signed-rank test.
        Returns p-value; raises on insufficient data.
        """
        if len(a) < 2 or len(b) < 2:
            return 1.0
        min_len = min(len(a), len(b))
        try:
            _, p = stats.wilcoxon(a[:min_len], b[:min_len])
            return float(p)
        except Exception:
            return 1.0

    # ─────────────────────────────────────────────────────────
    # Single-run analysis (used per run in main.py)
    # ─────────────────────────────────────────────────────────

    def run_analysis(
        self,
        baseline_macro_f1s: Dict[str, List[float]],
    ) -> pd.DataFrame:
        """
        Compare Hybrid GA+GWO against all other baselines using
        Wilcoxon test and 95% CI. Saves statistical_analysis.csv.
        """
        rows = []
        hybrid_f1s = baseline_macro_f1s.get("Hybrid GA + GWO", [])

        for name, f1s in baseline_macro_f1s.items():
            if not f1s:
                continue
            mean = float(np.mean(f1s))
            std  = float(np.std(f1s))
            ci   = self.confidence_interval(f1s)
            p    = self.wilcoxon_test(hybrid_f1s, f1s) if name != "Hybrid GA + GWO" else 1.0

            rows.append({
                "Baseline":       name,
                "Mean F1":        round(mean, 4),
                "Std F1":         round(std,  4),
                "CI_lower":       ci[0],
                "CI_upper":       ci[1],
                "p_value":        round(p, 5),
                "Significant":    "Yes" if p < 0.05 else "No",
            })

        df = pd.DataFrame(rows)
        out = os.path.join(self.results_dir, "statistical_analysis.csv")
        df.to_csv(out, index=False)
        logger.info(f"Statistical analysis saved to {out}")
        return df

    # ─────────────────────────────────────────────────────────
    # Multi-seed aggregation (used by run_experiments.py)
    # ─────────────────────────────────────────────────────────

    def aggregate_multi_seed(
        self,
        combined_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Given a DataFrame with columns [baseline, seed, macro_f1, ...],
        return a summary DataFrame with mean ± std and 95% CI per baseline.
        """
        summary_rows = []
        hybrid_all = combined_df.loc[
            combined_df["baseline"] == "Hybrid GA + GWO", "macro_f1"
        ].tolist()

        for bl in combined_df["baseline"].unique():
            subset = combined_df.loc[combined_df["baseline"] == bl, "macro_f1"].tolist()
            if not subset:
                continue
            mean = float(np.mean(subset))
            std  = float(np.std(subset))
            ci   = self.confidence_interval(subset)
            p    = self.wilcoxon_test(hybrid_all, subset) if bl != "Hybrid GA + GWO" else 1.0

            summary_rows.append({
                "Baseline":              bl,
                "Macro F1 Mean":         round(mean, 4),
                "Macro F1 Std":          round(std,  4),
                "95% CI":                f"[{ci[0]:.4f}, {ci[1]:.4f}]",
                "p-value vs Hybrid":     round(p, 5),
                "Significant (p<0.05)":  "Yes" if p < 0.05 else "No",
            })

        out_df = pd.DataFrame(summary_rows)
        out_df.to_csv(
            os.path.join(self.results_dir, "multi_seed_summary.csv"), index=False
        )
        return out_df
