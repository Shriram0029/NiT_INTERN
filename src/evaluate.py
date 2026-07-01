import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

class Evaluator:
    def evaluate(self, model, tokenizer, test_data, device):
        if not model or not test_data:
            return {"macro_f1": 0.0, "precision": 0.0, "recall": 0.0, "confusion_matrix": []}
            
        model.eval()
        texts = [item[0] for item in test_data]
        true_labels = [item[1] for item in test_data]
        
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            
        macro_f1 = f1_score(true_labels, preds, average='macro', zero_division=0)
        precision = precision_score(true_labels, preds, average='macro', zero_division=0)
        recall = recall_score(true_labels, preds, average='macro', zero_division=0)
        cm = confusion_matrix(true_labels, preds).tolist()
        
        return {
            "macro_f1": macro_f1,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": cm,
            "predictions": preds.tolist(),
            "true_labels": true_labels
        }
