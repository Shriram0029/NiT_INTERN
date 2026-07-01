import random
import logging
import torch
from transformers import pipeline
import nlpaug.augmenter.word as naw
from src.semantic_validator import SemanticValidator

logger = logging.getLogger(__name__)

class Augmentor:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device_id = 0 if self.device == 'cuda' else -1
        
        try:
            self.back_translation_aug = naw.BackTranslationAug(
                from_model_name='facebook/wmt19-en-de',
                to_model_name='facebook/wmt19-de-en',
                device=self.device
            )
            
            models = [
                self.back_translation_aug.model.src_model, 
                self.back_translation_aug.model.tgt_model
            ]
            
            for model in models:
                model.config.do_sample = True
                model.config.top_k = 50
                model.config.top_p = 0.95
                model.config.temperature = 0.8
        except Exception as e:
            logger.warning(f"Could not load nlpaug translation models: {e}")
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
            augmented_text = self.back_translation_aug.augment(text)
            if isinstance(augmented_text, list):
                return augmented_text[0]
            return augmented_text
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

    def generate(self, text, entities, policy):
        # [BT_Ratio, BERT_Ratio, Budget, MaskProbability, SemanticThreshold, EntityProtectionWeight, ChunkPriority, LR]
        p_bt, p_bert, budget, mask_prob, sem_thresh, ent_weight, _, _ = policy
        budget = int(round(budget))
        
        validator = SemanticValidator(semantic_threshold=sem_thresh, entity_weight=ent_weight)
        
        results = set()
        for _ in range(budget):
            if random.random() < p_bt:
                aug = self.back_translate(text)
            else:
                aug = self.bert_masking(text, mask_prob)
                
            if validator.validate(text, aug, entities):
                if aug != text:
                    results.add(aug)
        return list(results)
