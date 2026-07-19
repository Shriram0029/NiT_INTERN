import torch
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score, confusion_matrix, accuracy_score,
    roc_auc_score, average_precision_score
)
from sklearn.preprocessing import label_binarize

class Evaluator:
    def __init__(self, num_classes=5):
        self.num_classes = num_classes

    def evaluate(self, model, tokenizer, test_data, device):
        if not model or not test_data:
            return {
                "macro_f1": 0.0, "weighted_f1": 0.0, "precision": 0.0, "recall": 0.0, 
                "accuracy": 0.0, "per_class_f1": [], "per_class_precision": [],
                "per_class_recall": [], "confusion_matrix": [], "roc_auc": 0.0, "pr_auc": 0.0
            }
            
        texts = [item[0] for item in test_data]
        true_labels = [item[1] for item in test_data]
        
        try:
            X = tokenizer.transform(texts)
            probs = model.predict_proba(X)
            preds = model.predict(X)
        except Exception as e:
            # Fallback if model is not fitted or fails
            probs = np.zeros((len(texts), self.num_classes))
            preds = np.zeros(len(texts))

            
        accuracy = accuracy_score(true_labels, preds)
        macro_f1 = f1_score(true_labels, preds, average='macro', zero_division=0)
        weighted_f1 = f1_score(true_labels, preds, average='weighted', zero_division=0)
        precision = precision_score(true_labels, preds, average='macro', zero_division=0)
        recall = recall_score(true_labels, preds, average='macro', zero_division=0)
        
        per_class_f1 = f1_score(true_labels, preds, average=None, zero_division=0).tolist()
        per_class_prec = precision_score(true_labels, preds, average=None, zero_division=0).tolist()
        per_class_rec = recall_score(true_labels, preds, average=None, zero_division=0).tolist()
        
        # Ensure array size matches num_classes (if test set lacks some classes)
        full_per_class_f1 = [0.0] * self.num_classes
        full_per_class_prec = [0.0] * self.num_classes
        full_per_class_rec = [0.0] * self.num_classes
        for i, val in enumerate(per_class_f1): full_per_class_f1[i] = val
        for i, val in enumerate(per_class_prec): full_per_class_prec[i] = val
        for i, val in enumerate(per_class_rec): full_per_class_rec[i] = val
        
        cm = confusion_matrix(true_labels, preds, labels=range(self.num_classes)).tolist()
        
        roc_auc = 0.0
        pr_auc = 0.0
        
        try:
            # Binarize labels for multi-class ROC/PR
            y_bin = label_binarize(true_labels, classes=range(self.num_classes))
            if self.num_classes == 2:
                # Need 2 columns for ROC
                y_bin = np.hstack((1 - y_bin, y_bin))
                
            roc_auc = roc_auc_score(y_bin, probs, average='macro', multi_class='ovr')
            pr_auc = average_precision_score(y_bin, probs, average='macro')
        except Exception:
            pass # Fails if a class is completely missing in test set
        
        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "precision": precision,
            "recall": recall,
            "per_class_f1": full_per_class_f1,
            "per_class_precision": full_per_class_prec,
            "per_class_recall": full_per_class_rec,
            "confusion_matrix": cm,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "predictions": preds.tolist(),
            "true_labels": true_labels,
            "probs": probs.tolist()
        }
