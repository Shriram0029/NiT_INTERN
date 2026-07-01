import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW

logger = logging.getLogger(__name__)

class IncrementalClassifier:
    def __init__(self, num_classes=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            self.model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=num_classes)
            self.model.to(self.device)
            # Default LR, will be overridden by policy learning rate per chunk
            self.optimizer = AdamW(self.model.parameters(), lr=2e-5)
        except Exception as e:
            logger.error(f"Failed to load RoBERTa model: {e}")
            self.tokenizer = None
            self.model = None

    def train_on_batch(self, batch, learning_rate=2e-5):
        if not self.model or not batch:
            return 0.0
            
        # Update LR based on policy
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
            
        self.model.train()
        texts = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
        labels = torch.tensor(labels).to(self.device)
        
        self.optimizer.zero_grad()
        outputs = self.model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
