import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
from collections import Counter
from sklearn.metrics import confusion_matrix

class Visualizer:
    def __init__(self, out_dir="visualizations"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
            
    def plot_training_loss(self, loss_history, filename="training_loss.png"):
        if not loss_history: return
        plt.figure(figsize=(10,6))
        plt.plot(range(1, len(loss_history)+1), loss_history, label="Training Loss")
        plt.title("Training Loss Over Time")
        plt.xlabel("Chunk")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.out_dir, filename))
        plt.close()
        
    def plot_macro_f1(self, f1_history, filename="macro_f1.png"):
        if not f1_history: return
        plt.figure(figsize=(10,6))
        plt.plot(range(1, len(f1_history)+1), f1_history, label="Macro F1", color="green")
        plt.title("Test Macro F1 Over Time")
        plt.xlabel("Chunk")
        plt.ylabel("Macro F1")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.out_dir, filename))
        plt.close()

    def plot_faci_distribution(self, faci_scores, filename="faci_distribution.png"):
        if not faci_scores: return
        plt.figure(figsize=(10,6))
        sns.histplot(faci_scores, bins=20, kde=True)
        plt.title("FACI Score Distribution")
        plt.xlabel("FACI Scalar Score")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(self.out_dir, filename))
        plt.close()

    def plot_optimizer_convergence(self, fitness_history, filename="optimizer_convergence.png"):
        if not fitness_history: return
        plt.figure(figsize=(10,6))
        plt.plot(range(1, len(fitness_history)+1), fitness_history, label="Optimizer Best Fitness")
        plt.title("Hybrid Optimizer Convergence")
        plt.xlabel("Optimization Steps (Chunks)")
        plt.ylabel("Fitness")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.out_dir, filename))
        plt.close()
        
    def plot_policy_distribution(self, strategy_list, filename="policy_distribution.png"):
        if not strategy_list: return
        plt.figure(figsize=(8,6))
        sns.countplot(y=strategy_list)
        plt.title("Selected Augmentation Strategy Distribution")
        plt.xlabel("Count")
        plt.ylabel("Strategy")
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, filename))
        plt.close()
        
    def plot_replay_distribution(self, class_dist_dict, filename="replay_distribution.png"):
        if not class_dist_dict: return
        classes = list(class_dist_dict.keys())
        counts = list(class_dist_dict.values())
        plt.figure(figsize=(8,6))
        sns.barplot(x=classes, y=counts)
        plt.title("Final Replay Buffer Class Distribution")
        plt.xlabel("Class")
        plt.ylabel("Samples in Buffer")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, filename))
        plt.close()
        
    def plot_confusion_matrix(self, y_true, y_pred, classes, filename="confusion_matrix.png"):
        if not y_true or not y_pred: return
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title("Final Evaluation Confusion Matrix")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, filename))
        plt.close()
