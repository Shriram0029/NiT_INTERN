import random
import logging
import torch
from transformers import pipeline
from src.semantic_validator import SemanticValidator

logger = logging.getLogger(__name__)

class Augmentor:
    def __init__(self):
        self.device = 'cpu'
        self.en_de = None
        self.de_en = None
        self.fill_mask = None
        self.validator = SemanticValidator()

    def back_translate(self, text):
        return text + " [BT]"

    def bert_masking(self, text, mask_prob):
        return text + " [BM]"

    def generate(self, text, label, chunk_id, prediction, dynamic_threshold=None):
        strategy = prediction.get("strategy", "No Augmentation")
        budget = prediction.get("budget", 0)
        
        results = set()
        
        if strategy == "No Augmentation" or budget <= 0:
            return []
            
        mask_prob = prediction.get("mask_prob", 0.15)
        
        entities = []
        for token in ["[OTP]", "[CARD]", "[ACCOUNT]", "[UPI]", "[IFSC]"]:
            if token in text:
                entities.append(token)
                
        for _ in range(budget):
            aug = text
            success = False
            
            for attempt in range(3):
                if strategy == "Back Translation":
                    aug = self.back_translate(text)
                elif strategy == "BERT Contextual":
                    aug = self.bert_masking(text, mask_prob)
                elif strategy == "Hybrid":
                    if random.random() < 0.5:
                        aug = self.back_translate(text)
                    else:
                        aug = self.bert_masking(text, mask_prob)
                        
                # Semantic Validation
                is_valid, reason = self.validator.validate_and_log(
                    chunk_id=chunk_id,
                    original_text=text,
                    augmented_text=aug,
                    original_label=label,
                    augmented_label=label,
                    entities=entities,
                    existing_samples=list(results),
                    method=strategy,
                    dynamic_threshold=dynamic_threshold
                )
                
                if aug == text:
                    continue
                    
                if is_valid:
                    results.add(aug)
                    success = True
                    break
                    
            if not success:
                logger.debug(f"Augmentation failed. Fallback.")
                
        return list(results)
