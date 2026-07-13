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

    def back_translate(self, text):
        if not self.back_translation_aug:
            return text
        try:
            de = self.en_de(text)[0]['translation_text']
            en = self.de_en(de)[0]['translation_text']
            return en
        except Exception:
            return text

    def bert_masking(self, text, mask_prob):
        if not self.fill_mask:
            return text
        
        words = text.split()
        if len(words) < 5:
            return text
            
        protected_tokens = ["[CARD]", "[OTP]"]
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

    def get_strategy(self, p_bt, p_bert, budget):
        if budget < 1 or (p_bt < 0.2 and p_bert < 0.2):
            return "No Augmentation"
        elif p_bt > p_bert + 0.3:
            return "Back Translation"
        elif p_bert > p_bt + 0.3:
            return "BERT Contextual"
        else:
            return "Back Translation + BERT"

    def generate(self, text, entities, policy):
        # [BT_Ratio, BERT_Ratio, Budget, MaskProbability, SemanticThreshold, EntityProtectionWeight, ChunkPriority, LR]
        p_bt, p_bert, budget, mask_prob, sem_thresh, ent_weight, _, _ = policy
        budget = int(round(budget))
        
        validator = SemanticValidator(semantic_threshold=sem_thresh, entity_weight=ent_weight)
        
        strategy = self.get_strategy(p_bt, p_bert, budget)
        results = set()
        
        if strategy == "No Augmentation":
            return []
            
        for _ in range(budget):
            if strategy == "Back Translation":
                aug = self.back_translate(text)
            elif strategy == "BERT Contextual":
                aug = self.bert_masking(text, mask_prob)
            else:
                if random.random() < 0.5:
                    aug = self.back_translate(text)
                else:
                    aug = self.bert_masking(text, mask_prob)
                    
            if validator.validate(text, aug, entities, existing_samples=list(results)):
                if aug != text:
                    results.add(aug)
                    
        return list(results)
