"""
src/visualizer.py — IEEE/Elsevier-standard publication figures (300 DPI)
Nature-Inspired Augmentation Selection Pipeline
"""

import os
import warnings
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")

# ── Global IEEE figure style ──────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":           300,
    "savefig.dpi":          300,
    "font.family":          "serif",
    "font.serif":           ["Times New Roman", "DejaVu Serif"],
    "font.size":            11,
    "axes.titlesize":       12,
    "axes.labelsize":       11,
    "xtick.labelsize":      9,
    "ytick.labelsize":      9,
    "legend.fontsize":      9,
    "axes.grid":            True,
    "grid.linestyle":       "--",
    "grid.alpha":           0.4,
    "lines.linewidth":      1.8,
    "figure.figsize":       (6.5, 4.0),   # IEEE single-column width ≈ 3.5″; double ≈ 7″
    "figure.constrained_layout.use": True,
})

# Colorblind-friendly palette (Okabe–Ito)
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
           "#56B4E9", "#F0E442", "#000000"]
sns.set_palette(PALETTE)


class Visualizer:
    """Generate publication-quality figures for the augmentation framework."""

    def __init__(self, out_dir: str = "visualizations") -> None:
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def _save(self, filename: str) -> None:
        path = os.path.join(self.out_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches="tight", format="png")
        plt.close()
        print(f"[Visualizer] Saved: {path}")

    # ─────────────────────────────────────────────────────────
    # Core training curves
    # ─────────────────────────────────────────────────────────

    def plot_training_loss(
        self,
        loss_history: List[float],
        filename: str = "training_loss.png",
    ) -> None:
        if not loss_history:
            return
        fig, ax = plt.subplots()
        ax.plot(range(1, len(loss_history) + 1), loss_history,
                color=PALETTE[0], label="Training Loss")
        ax.set_title("Training Loss per Data Chunk")
        ax.set_xlabel("Chunk Index")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.legend()
        self._save(filename)

    def plot_validation_loss(
        self,
        val_loss: List[float],
        filename: str = "validation_loss.png",
    ) -> None:
        if not val_loss:
            return
        fig, ax = plt.subplots()
        ax.plot(range(1, len(val_loss) + 1), val_loss,
                color=PALETTE[1], linestyle="--", label="Validation Loss")
        ax.set_title("Validation Loss per Data Chunk")
        ax.set_xlabel("Chunk Index")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.legend()
        self._save(filename)

    def plot_macro_f1(
        self,
        f1_history: List[float],
        filename: str = "macro_f1.png",
    ) -> None:
        if not f1_history:
            return
        fig, ax = plt.subplots()
        ax.plot(range(1, len(f1_history) + 1), f1_history,
                color=PALETTE[2], label="Macro F1")
        ax.set_title("Macro F1 Score per Data Chunk")
        ax.set_xlabel("Chunk Index")
        ax.set_ylabel("Macro F1")
        ax.set_ylim(0.0, 1.0)
        ax.legend()
        self._save(filename)

    def plot_precision_recall(
        self,
        precision: List[float],
        recall: List[float],
        filename: str = "precision_recall.png",
    ) -> None:
        if not precision or not recall:
            return
        x = range(1, len(precision) + 1)
        fig, ax = plt.subplots()
        ax.plot(x, precision, color=PALETTE[3], label="Macro Precision")
        ax.plot(x, recall,    color=PALETTE[4], linestyle="--", label="Macro Recall")
        ax.set_title("Precision & Recall per Data Chunk")
        ax.set_xlabel("Chunk Index")
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.0)
        ax.legend()
        self._save(filename)

    # ─────────────────────────────────────────────────────────
    # Optimiser & FACI plots
    # ─────────────────────────────────────────────────────────

    def plot_optimizer_convergence(
        self,
        fitness_history: List[float],
        filename: str = "optimizer_convergence.png",
    ) -> None:
        if not fitness_history:
            return
        fig, ax = plt.subplots()
        ax.plot(range(1, len(fitness_history) + 1), fitness_history,
                color=PALETTE[5], label="Hybrid GA-GWO Fitness")
        ax.set_title("Hybrid GA-GWO Convergence")
        ax.set_xlabel("Optimisation Step (Chunk)")
        ax.set_ylabel("Best Fitness Score")
        ax.legend()
        self._save(filename)

    def plot_faci_distribution(
        self,
        faci_scores: List[float],
        filename: str = "faci_distribution.png",
    ) -> None:
        if not faci_scores:
            return
        fig, ax = plt.subplots()
        sns.histplot(faci_scores, bins=15, kde=True, ax=ax,
                     color=PALETTE[0], edgecolor="white")
        ax.set_title("FACI Score Distribution Across Chunks")
        ax.set_xlabel("FACI Scalar Score")
        ax.set_ylabel("Frequency")
        self._save(filename)

    # ─────────────────────────────────────────────────────────
    # Policy & buffer plots
    # ─────────────────────────────────────────────────────────

    def plot_policy_distribution(
        self,
        strategy_list: List[str],
        filename: str = "policy_distribution.png",
    ) -> None:
        if not strategy_list:
            return
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        unique = sorted(set(strategy_list))
        counts = [strategy_list.count(s) for s in unique]
        bars = ax.barh(unique, counts, color=PALETTE[:len(unique)])
        ax.bar_label(bars, padding=3, fontsize=9)
        ax.set_title("Augmentation Strategy Selection Distribution")
        ax.set_xlabel("Count")
        ax.set_ylabel("Strategy")
        self._save(filename)

    def plot_replay_distribution(
        self,
        class_dist_dict: Dict[str, int],
        filename: str = "replay_distribution.png",
    ) -> None:
        if not class_dist_dict:
            return
        classes = [str(k) for k in class_dist_dict.keys()]
        counts  = list(class_dist_dict.values())
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        bars = ax.bar(classes, counts, color=PALETTE[:len(classes)])
        ax.bar_label(bars, padding=3, fontsize=9)
        ax.set_title("Final Replay Buffer Class Distribution")
        ax.set_xlabel("Class")
        ax.set_ylabel("Samples in Buffer")
        plt.xticks(rotation=30, ha="right")
        self._save(filename)

    def plot_class_distribution(
        self,
        label_dist: Dict[str, int],
        filename: str = "class_distribution.png",
    ) -> None:
        if not label_dist:
            return
        labels = list(label_dist.keys())
        counts = list(label_dist.values())
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        bars = ax.barh(labels, counts, color=PALETTE[:len(labels)])
        ax.bar_label(bars, padding=3, fontsize=9)
        ax.set_title("Dataset Class Distribution")
        ax.set_xlabel("Sample Count")
        self._save(filename)

    # ─────────────────────────────────────────────────────────
    # Comparison plots
    # ─────────────────────────────────────────────────────────

    def plot_baseline_comparison(
        self,
        baseline_f1s: Dict[str, float],
        filename: str = "baseline_comparison.png",
    ) -> None:
        if not baseline_f1s:
            return
        names  = list(baseline_f1s.keys())
        values = list(baseline_f1s.values())
        colors = [
            PALETTE[2] if "Hybrid" in n else PALETTE[0]
            for n in names
        ]
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(names, values, color=colors)
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
        ax.set_title("Macro F1 — Baseline Comparison")
        ax.set_ylabel("Macro F1 Score")
        ax.set_ylim(0.0, 1.05)
        plt.xticks(rotation=30, ha="right")
        self._save(filename)

    def plot_runtime_comparison(
        self,
        runtime_dict: Dict[str, float],
        filename: str = "runtime_comparison.png",
    ) -> None:
        if not runtime_dict:
            return
        names  = list(runtime_dict.keys())
        values = list(runtime_dict.values())
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(names, values, color=PALETTE[:len(names)])
        ax.bar_label(bars, fmt="%.1fs", padding=3, fontsize=8)
        ax.set_title("Average Runtime per Chunk — Baseline Comparison")
        ax.set_ylabel("Seconds per Chunk")
        plt.xticks(rotation=30, ha="right")
        self._save(filename)

    def plot_ablation_comparison(
        self,
        ablation_df,
        filename: str = "ablation_comparison.png",
    ) -> None:
        if ablation_df is None or ablation_df.empty:
            return
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(
            ablation_df["Configuration"],
            ablation_df["Macro F1 Mean"],
            xerr=ablation_df.get("Macro F1 Std", None),
            color=PALETTE[0],
            capsize=4,
        )
        ax.set_title("Ablation Study — Macro F1 Impact")
        ax.set_xlabel("Macro F1 Mean")
        ax.set_xlim(0.0, 1.0)
        self._save(filename)

    # ─────────────────────────────────────────────────────────
    # Confusion matrix
    # ─────────────────────────────────────────────────────────

    def plot_confusion_matrix(
        self,
        y_true: List[int],
        y_pred: List[int],
        classes: List[str],
        filename: str = "confusion_matrix.png",
    ) -> None:
        if not y_true or not y_pred:
            return
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title("Confusion Matrix — Hybrid GA+GWO (Final Evaluation)")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        plt.xticks(rotation=30, ha="right")
        self._save(filename)
