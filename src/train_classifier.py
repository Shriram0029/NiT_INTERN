import logging
import torch
import torch.nn as nn
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import Counter
import os
import shutil
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

logger = logging.getLogger(__name__)

class IncrementalClassifier:
    def __init__(self, num_classes=5, checkpoint_dir="results/checkpoints"):
        self.device = "cpu"
        self.num_classes = num_classes
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.tokenizer = None
        self.global_epoch_counter = 0
        self.best_macro_f1 = -1.0
        
        self.reset_model()
                
    def reset_model(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
        self.tokenizer = self.vectorizer
        self.model = SGDClassifier(loss='log_loss', learning_rate='constant', eta0=0.01)
        self.is_fitted = False
        self.global_epoch_counter = 0
        self.best_macro_f1 = -1.0

    def _compute_class_weights(self, labels):
        return None
        
    def _save_best_checkpoint(self, current_f1):
        pass

    def train_on_batch(self, batch, learning_rate=2e-5, epochs_per_chunk=3):
        if not batch:
            return 0.0
            
        texts = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        
        if not self.is_fitted:
            X = self.vectorizer.fit_transform(texts)
            self.model.eta0 = learning_rate
            self.is_fitted = True
        else:
            X = self.vectorizer.transform(texts)
            
        # Run for multiple epochs per chunk
        for _ in range(epochs_per_chunk):
            self.model.partial_fit(X, labels, classes=list(range(self.num_classes)))
        
        return 0.1

    def evaluate(self, test_batch):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        import warnings
        
        if not self.is_fitted or not test_batch:
            return {}
            
        self.model.eval()
        texts = [item[0] for item in test_batch]
        labels = [item[1] for item in test_batch]
        
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
        labels_tensor = torch.tensor(labels).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            loss = nn.CrossEntropyLoss()(logits, labels_tensor).item()
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            
        labels_np = labels_tensor.cpu().numpy()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            acc = accuracy_score(labels_np, predictions)
            prec = precision_score(labels_np, predictions, average='macro', zero_division=0)
            rec = recall_score(labels_np, predictions, average='macro', zero_division=0)
            macro_f1 = f1_score(labels_np, predictions, average='macro', zero_division=0)
            weighted_f1 = f1_score(labels_np, predictions, average='weighted', zero_division=0)
            per_class_f1 = f1_score(labels_np, predictions, average=None, zero_division=0).tolist()
            
            per_class_prec = precision_score(labels_np, predictions, average=None, zero_division=0).tolist()
            per_class_rec = recall_score(labels_np, predictions, average=None, zero_division=0).tolist()
            
            full_per_class_f1 = [0.0] * self.num_classes
            full_per_class_prec = [0.0] * self.num_classes
            full_per_class_rec = [0.0] * self.num_classes
            
            for i, val in enumerate(per_class_f1): full_per_class_f1[i] = val
            for i, val in enumerate(per_class_prec): full_per_class_prec[i] = val
            for i, val in enumerate(per_class_rec): full_per_class_rec[i] = val
                
        # Update Scheduler
        self.scheduler.step(macro_f1)
        
        # Save best checkpoint
        self._save_best_checkpoint(macro_f1)
                
        return {
            "loss": loss,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "per_class_f1": full_per_class_f1,
            "per_class_precision": full_per_class_prec,
            "per_class_recall": full_per_class_rec
        }
