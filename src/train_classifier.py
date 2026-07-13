import logging
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from collections import Counter

logger = logging.getLogger(__name__)

class IncrementalClassifier:
    def __init__(self, num_classes=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            self.model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=num_classes)
            self.model.to(self.device)
            self.optimizer = AdamW(self.model.parameters(), lr=2e-5)
            # Mixed Precision setup
            self.scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
        except Exception as e:
            logger.error(f"Failed to load RoBERTa model: {e}")
            self.tokenizer = None
            self.model = None

    def _compute_class_weights(self, labels):
        counts = Counter(labels)
        if not counts:
            return torch.ones(self.num_classes).to(self.device)
        total = sum(counts.values())
        # Inverse frequency weighting
        weights = [total / max(counts.get(i, 1), 1) for i in range(self.num_classes)]
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        # Normalize weights
        weights = weights / weights.sum() * self.num_classes
        return weights

    def train_on_batch(self, batch, learning_rate=2e-5, epochs_per_chunk=3):
        if not self.model or not batch:
            return 0.0
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
            
        self.model.train()
        texts = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        
        class_weights = self._compute_class_weights(labels)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
        labels_tensor = torch.tensor(labels).to(self.device)
        
        best_loss = float('inf')
        final_loss = 0.0
        
        # Mini-epoch loop for early stopping inside chunk
        for epoch in range(epochs_per_chunk):
            self.optimizer.zero_grad()
            
            # Mixed Precision Forward
            with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = self.model(**inputs)
                logits = outputs.logits
                loss = loss_fn(logits, labels_tensor)
                
            # Mixed Precision Backward
            self.scaler.scale(loss).backward()
            
            # Gradient Clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            final_loss = loss.item()
            
            # Early Stopping Check
            if final_loss < 0.01:
                break
                
            if final_loss < best_loss:
                best_loss = final_loss
                
        return final_loss

