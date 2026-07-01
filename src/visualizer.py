import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
from math import pi

class Visualizer:
    def __init__(self, out_dir="visualizations"):
        self.out_dir = out_dir
        for subdir in ["training", "optimizer", "graph", "faci", "reports"]:
            os.makedirs(os.path.join(self.out_dir, subdir), exist_ok=True)
        
    def plot_training_metrics(self, history, title="Training Metrics", filename="training_loss.png"):
        if not history: return
        epochs = range(1, len(history) + 1)
        plt.figure(figsize=(10,6))
        plt.plot(epochs, history, label="Loss", marker='o')
        plt.title(title)
        plt.xlabel("Chunk / Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.out_dir, "training", filename))
        plt.close()
        
    def plot_faci_distribution(self, faci_scores, filename="faci_distribution.png"):
        if not faci_scores: return
        plt.figure(figsize=(10,6))
        sns.histplot(faci_scores, bins=10, kde=True)
        plt.title("FACI Score Distribution")
        plt.xlabel("FACI Score")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(self.out_dir, "faci", filename))
        plt.close()
        
    def plot_faci_radar(self, faci_dict, filename="faci_radar.png"):
        if not faci_dict: return
        categories = list(faci_dict.keys())
        categories = [c for c in categories if c not in ["scalar", "recommended_budget"]]
        values = [faci_dict[c] for c in categories]
        
        if not categories: return
        
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

    def plot_ga_metrics(self, diversity_hist, mutation_hist, filename="ga_metrics.png"):
        if not diversity_hist or not mutation_hist: return
        fig, ax1 = plt.subplots(figsize=(10,6))
        
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Population Diversity', color='tab:blue')
        ax1.plot(diversity_hist, color='tab:blue', label="Diversity")
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Mutation Rate', color='tab:orange')
        ax2.plot(mutation_hist, color='tab:orange', linestyle='--', label="Mutation Rate")
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        
        fig.tight_layout()
        plt.title("GA Adaptive Mutation & Diversity")
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
        plt.xlabel("Iterations")
        plt.ylabel("Fitness")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.out_dir, "optimizer", filename))
        plt.close()
        
    def plot_confusion_matrix(self, cm, classes, filename="confusion_matrix.png"):
        if not cm: return
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title("Confusion Matrix")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(os.path.join(self.out_dir, "reports", filename))
        plt.close()
