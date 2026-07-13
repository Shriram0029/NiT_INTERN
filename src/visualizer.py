import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
from math import pi
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

class Visualizer:
    def __init__(self, out_dir="visualizations"):
        self.out_dir = out_dir
        for subdir in ["training", "optimizer", "graph", "faci", "reports", "policy", "replay"]:
            os.makedirs(os.path.join(self.out_dir, subdir), exist_ok=True)
            
    # --- Training ---
    def plot_training_metrics(self, history_dict, filename="training_metrics.png"):
        if not history_dict: return
        epochs = range(1, len(history_dict.get('loss', [])) + 1)
        if not epochs: return
        
        plt.figure(figsize=(12,8))
        for key, values in history_dict.items():
            if values:
                plt.plot(epochs, values, label=key.capitalize(), marker='o')
        plt.title("Training Metrics Over Chunks")
        plt.xlabel("Chunk")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.out_dir, "training", filename))
        plt.close()
        
    # --- FACI ---
    def plot_faci_distribution(self, faci_scores, filename="faci_distribution.png"):
        if not faci_scores: return
        plt.figure(figsize=(10,6))
        sns.histplot(faci_scores, bins=20, kde=True)
        plt.title("FACI Score Distribution")
        plt.xlabel("FACI Score")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(self.out_dir, "faci", filename))
        plt.close()
        
    def plot_faci_radar(self, faci_dict, filename="faci_radar.png"):
        if not faci_dict: return
        categories = [c for c in faci_dict.keys() if isinstance(faci_dict[c], (int, float)) and c not in ["scalar", "recommended_budget", "class_imbalance_factor"]]
        if not categories: return
        values = [faci_dict[c] for c in categories]
        
        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        values += values[:1]
        
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        plt.xticks(angles[:-1], categories, color='grey', size=8)
        ax.plot(angles, values, linewidth=1, linestyle='solid')
        ax.fill(angles, values, 'b', alpha=0.1)
        plt.title("Multi-Dimensional FACI")
        plt.savefig(os.path.join(self.out_dir, "faci", filename))
        plt.close()
        
    def plot_faci_correlation(self, faci_data_list, filename="faci_correlation.png"):
        if not faci_data_list: return
        df = pd.DataFrame(faci_data_list)
        cols = [c for c in df.columns if df[c].dtype in [np.float64, np.float32, int] and c not in ["scalar", "recommended_budget"]]
        if not cols: return
        
        corr = df[cols].corr()
        plt.figure(figsize=(10,8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("FACI Features Correlation Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "faci", filename))
        plt.close()
        
    def plot_faci_feature_importance(self, importance_dict, filename="faci_importance.png"):
        if not importance_dict: return
        labels = list(importance_dict.keys())
        values = list(importance_dict.values())
        
        plt.figure(figsize=(10,6))
        sns.barplot(x=values, y=labels)
        plt.title("FACI Feature Importance")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "faci", filename))
        plt.close()

    # --- Optimizer ---
    def plot_ga_metrics(self, ga_metrics, filename="ga_metrics.png"):
        if not ga_metrics: return
        div = ga_metrics.get("diversity_history", [])
        mut = ga_metrics.get("mutation_history", [])
        fit = ga_metrics.get("fitness_history", [])
        
        if not div: return
        
        fig, ax1 = plt.subplots(figsize=(10,6))
        
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Population Diversity & Elite Fitness', color='tab:blue')
        ax1.plot(div, color='tab:blue', label="Diversity", linestyle='-')
        if fit: ax1.plot(fit, color='tab:green', label="Elite Fitness", linestyle='-.')
        ax1.tick_params(axis='y')
        ax1.legend(loc='upper left')
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Mutation Rate', color='tab:orange')
        ax2.plot(mut, color='tab:orange', linestyle='--', label="Mutation Rate")
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        ax2.legend(loc='upper right')
        
        fig.tight_layout()
        plt.title("GA Adaptive Metrics")
        plt.savefig(os.path.join(self.out_dir, "optimizer", filename))
        plt.close()
        
    def plot_gwo_metrics(self, gwo_metrics, filename="gwo_metrics.png"):
        if not gwo_metrics: return
        
        alpha = gwo_metrics.get("alpha_fitness", [])
        beta = gwo_metrics.get("beta_fitness", [])
        delta = gwo_metrics.get("delta_fitness", [])
        
        if not alpha: return
        
        plt.figure(figsize=(10,6))
        plt.plot(alpha, label="Alpha Fitness", marker='o')
        plt.plot(beta, label="Beta Fitness", marker='x')
        plt.plot(delta, label="Delta Fitness", marker='^')
        plt.title("GWO Pack Fitness Evolution")
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.out_dir, "optimizer", filename))
        plt.close()
        
    def plot_optimizer_convergence(self, ga_history, gwo_history, filename="optimizer_convergence.png"):
        if not ga_history and not gwo_history: return
        plt.figure(figsize=(10,6))
        if ga_history:
            plt.plot(range(len(ga_history)), ga_history, label="GA Elite Fitness", linestyle='--')
        if gwo_history:
            plt.plot(range(len(ga_history)-1, len(ga_history)-1+len(gwo_history)), gwo_history, label="GWO Alpha Fitness", marker='x')
        plt.title("Hybrid Optimization Convergence")
        plt.xlabel("Iterations (GA -> GWO)")
        plt.ylabel("Fitness")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.out_dir, "optimizer", filename))
        plt.close()
        
    # --- Policy ---
    def plot_strategy_distribution(self, strategy_list, filename="strategy_dist.png"):
        if not strategy_list: return
        plt.figure(figsize=(8,6))
        sns.countplot(y=strategy_list)
        plt.title("Selected Augmentation Strategy Distribution")
        plt.xlabel("Count")
        plt.ylabel("Strategy")
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "policy", filename))
        plt.close()
        
    # --- Replay Buffer ---
    def plot_replay_distribution(self, class_dist_dict, filename="replay_dist.png"):
        if not class_dist_dict: return
        classes = list(class_dist_dict.keys())
        counts = list(class_dist_dict.values())
        
        plt.figure(figsize=(8,6))
        sns.barplot(x=classes, y=counts)
        plt.title("Replay Buffer Class Distribution")
        plt.xlabel("Class")
        plt.ylabel("Samples in Buffer")
        plt.savefig(os.path.join(self.out_dir, "replay", filename))
        plt.close()
        
    # --- Classification ---
    def plot_confusion_matrix(self, cm, classes, filename="confusion_matrix.png"):
        if not cm: return
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title("Confusion Matrix")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "reports", filename))
        plt.close()

