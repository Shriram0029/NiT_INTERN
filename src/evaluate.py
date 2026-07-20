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
                "accuracy": 0.0, "per_class_f1": [0.0]*self.num_classes, 
                "per_class_precision": [0.0]*self.num_classes,
                "per_class_recall": [0.0]*self.num_classes, "confusion_matrix": [], 
                "roc_auc": 0.0, "pr_auc": 0.0, "predictions": [], "true_labels": [], "probs": []
            }
            
        texts = [item[0] for item in test_data]
        true_labels = [item[1] for item in test_data]
        
        model.eval()
        all_preds = []
        all_probs = []
        
        batch_size = 32
        val_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_labels = true_labels[i:i+batch_size]
                inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
                labels_tensor = torch.tensor(batch_labels).to(device)
                
                outputs = model(**inputs, labels=labels_tensor)
                val_loss += outputs.loss.item()
                num_batches += 1
                
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                
        val_loss = val_loss / max(1, num_batches)
        preds = np.array(all_preds)
        probs = np.array(all_probs)
        
        accuracy = accuracy_score(true_labels, preds)
        macro_f1 = f1_score(true_labels, preds, average='macro', zero_division=0)
        weighted_f1 = f1_score(true_labels, preds, average='weighted', zero_division=0)
        precision = precision_score(true_labels, preds, average='macro', zero_division=0)
        recall = recall_score(true_labels, preds, average='macro', zero_division=0)
        
        per_class_f1 = f1_score(true_labels, preds, average=None, zero_division=0).tolist()
        per_class_prec = precision_score(true_labels, preds, average=None, zero_division=0).tolist()
        per_class_rec = recall_score(true_labels, preds, average=None, zero_division=0).tolist()
        
        # Ensure array size matches num_classes
        full_per_class_f1 = [0.0] * self.num_classes
        full_per_class_prec = [0.0] * self.num_classes
        full_per_class_rec = [0.0] * self.num_classes
        for i, val in enumerate(per_class_f1): 
            if i < self.num_classes: full_per_class_f1[i] = val
        for i, val in enumerate(per_class_prec): 
            if i < self.num_classes: full_per_class_prec[i] = val
        for i, val in enumerate(per_class_rec): 
            if i < self.num_classes: full_per_class_rec[i] = val
        
        cm = confusion_matrix(true_labels, preds, labels=range(self.num_classes)).tolist()
        
        roc_auc = 0.0
        pr_auc = 0.0
        
        try:
            y_bin = label_binarize(true_labels, classes=range(self.num_classes))
            if self.num_classes == 2:
                y_bin = np.hstack((1 - y_bin, y_bin))
                
            roc_auc = roc_auc_score(y_bin, probs, average='macro', multi_class='ovr')
            pr_auc = average_precision_score(y_bin, probs, average='macro')
        except Exception:
            pass
        
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
            "probs": probs.tolist(),
            "val_loss": val_loss
        }
