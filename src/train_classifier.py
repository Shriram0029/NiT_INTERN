import logging
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from torch.optim import AdamW
import os
import numpy as np
from collections import Counter

logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        # alpha should be a tensor of weights
        self.alpha = alpha 

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class IncrementalClassifier:
    def __init__(self, num_classes=5, checkpoint_dir="results/checkpoints", total_steps=1000):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_classes = num_classes
        self.checkpoint_dir = checkpoint_dir
        self.total_steps = total_steps
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        self.current_step = 0
        
        self.reset_model(total_steps=total_steps)
                
    def reset_model(self, total_steps=1000):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", 
            num_labels=self.num_classes
        ).to(self.device)
        self.optimizer = None
        self.scheduler = None
        self.is_fitted = False
        self.current_step = 0
        self.total_steps = total_steps

    def train_on_batch(self, batch, learning_rate=2e-5, epochs_per_chunk=3):
        if not batch:
            return 0.0, "None", [], 0.0, []
            
        texts = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        
        if self.optimizer is None:
            self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
            warmup_steps = int(0.10 * self.total_steps)
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=warmup_steps, 
                num_training_steps=self.total_steps
            )
            
        self.model.train()
        total_loss = 0.0
        
        # Analyze class frequencies to detect imbalance
        class_counts = Counter(labels)
        total_samples = len(labels)
        
        # Calculate class weights
        weights = [1.0] * self.num_classes
        min_count = float('inf')
        max_count = 0
        for i in range(self.num_classes):
            c = class_counts.get(i, 0)
            if c > 0:
                weights[i] = total_samples / (self.num_classes * c)
                if c < min_count: min_count = c
                if c > max_count: max_count = c
            else:
                weights[i] = 1.0 # default weight for absent classes in this batch
                
        # Normalize weights
        weight_sum = sum(weights)
        if weight_sum > 0:
            weights = [w / weight_sum * self.num_classes for w in weights]
            
        # Detect Imbalance and choose Loss Function
        imbalance_ratio = max_count / max(min_count, 1)
        selected_loss_fn_name = "Standard CrossEntropy"
        gamma = 0.0
        alpha_list = []
        
        if imbalance_ratio > 3.0:
            selected_loss_fn_name = "Focal Loss"
            gamma = 2.0
            alpha_list = weights
            alpha_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)
            criterion = FocalLoss(alpha=alpha_tensor, gamma=gamma)
        elif imbalance_ratio > 1.5:
            selected_loss_fn_name = "Weighted CrossEntropy"
            alpha_list = weights
            alpha_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=alpha_tensor)
        else:
            criterion = nn.CrossEntropyLoss()
        
        batch_size = 16
        num_batches = len(texts) // batch_size + (1 if len(texts) % batch_size != 0 else 0)
        
        last_lr = self.scheduler.get_last_lr()[0]
        
        for _ in range(epochs_per_chunk):
            epoch_loss = 0.0
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(texts))
                
                batch_texts = texts[start_idx:end_idx]
                batch_labels = labels[start_idx:end_idx]
                
                inputs = self.tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=128, 
                    return_tensors="pt"
                ).to(self.device)
                
                labels_tensor = torch.tensor(batch_labels).to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(**inputs)
                logits = outputs.logits
                loss = criterion(logits, labels_tensor)
                
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.current_step += 1
                
                last_lr = self.scheduler.get_last_lr()[0]
                
                epoch_loss += loss.item()
                
            total_loss += (epoch_loss / max(1, num_batches))
            
        avg_loss = total_loss / epochs_per_chunk
        return avg_loss, selected_loss_fn_name, alpha_list, gamma, last_lr
