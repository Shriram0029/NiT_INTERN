import random
import logging
import torch
from transformers import pipeline
from src.semantic_validator import SemanticValidator

logger = logging.getLogger(__name__)

class Augmentor:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device_id = 0 if self.device == 'cuda' else -1
        
        try:
            self.en_de = pipeline("translation", model="Helsinki-NLP/opus-mt-en-de", device=self.device_id)
            self.de_en = pipeline("translation", model="Helsinki-NLP/opus-mt-de-en", device=self.device_id)
            self.back_translation_aug = True
        except Exception as e:
            logger.warning(f"Could not load HuggingFace translation models: {e}")
            self.back_translation_aug = None
            
        try:
            self.fill_mask = pipeline("fill-mask", model="bert-base-uncased", device=self.device_id)
        except Exception as e:
            logger.warning(f"Could not load BERT model: {e}")
            self.fill_mask = None
            
        self.validator = SemanticValidator(semantic_threshold=0.8, entity_weight=1.0)

    def back_translate(self, text):
        if not self.back_translation_aug:
            return text
            
        # Truncate to avoid max_length CUDA assert in Helsinki-NLP
        words = text.split()
        if len(words) > 350:
            text = " ".join(words[:350])
            
        try:
            de = self.en_de(text, truncation=True, max_length=512)[0]['translation_text']
            en = self.de_en(de, truncation=True, max_length=512)[0]['translation_text']
            return en
        except Exception as e:
            logger.warning(f"Back translation failed: {e}")
            return text

    def bert_masking(self, text, mask_prob):
        if not self.fill_mask:
            return text
        
        words = text.split()
        if len(words) > 350:
            words = words[:350]
            text = " ".join(words)
            
        if len(words) < 5:
            return text
            
        protected_tokens = ["[CARD]", "[OTP]", "xxxx"]
        mask_candidates = [i for i, w in enumerate(words) if not any(p in w for p in protected_tokens) and w.isalpha()]
        
        if not mask_candidates:
            return text
            
        num_mask = max(1, int(len(words) * mask_prob))
        if num_mask > len(mask_candidates):
            num_mask = len(mask_candidates)
            
        to_mask = random.sample(mask_candidates, num_mask)
        for idx in to_mask:
            words[idx] = "[MASK]"
            
        masked_text = " ".join(words)
        
        try:
            results = self.fill_mask(masked_text)
            if isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], list):
                    return text
                return results[0]['sequence']
            return text
        except Exception:
            return text

    def generate(self, text, label, chunk_id, prediction):
        strategy = prediction.get("strategy", "No Augmentation")
        budget = prediction.get("budget", 0)
        
        results = set()
        
        if strategy == "No Augmentation" or budget <= 0:
            return []
            
        # Hardcode mask prob for now, or could extract from policy if we had it
        mask_prob = 0.15 
        
        entities = []
        if "[OTP]" in text: entities.append("[OTP]")
        if "[CARD]" in text: entities.append("[CARD]")
        
        for _ in range(budget):
            aug = text
            if strategy == "Back Translation":
                aug = self.back_translate(text)
            elif strategy == "BERT Contextual":
                aug = self.bert_masking(text, mask_prob)
            elif strategy == "Hybrid":
                if random.random() < 0.5:
                    aug = self.back_translate(text)
                else:
                    aug = self.bert_masking(text, mask_prob)
                    
            is_valid, reason = self.validator.validate_and_log(
                chunk_id=chunk_id,
                original_text=text,
                augmented_text=aug,
                original_label=label,
                augmented_label=label,
                entities=entities,
                existing_samples=list(results)
            )
            
            if is_valid and aug != text:
                results.add(aug)
                
        return list(results)
