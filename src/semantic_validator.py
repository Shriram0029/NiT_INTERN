class SemanticValidator:
    def __init__(self, semantic_threshold=0.8, entity_weight=1.0):
        self.semantic_threshold = semantic_threshold
        self.entity_weight = entity_weight
        
        # Simple list of banking entities to check for hallucinations
        self.banking_entities = ["bank", "credit card", "account", "loan", "mortgage", "debit", "transaction", "payment", "fund", "wire", "transfer"]
        
    def validate(self, original_text, augmented_text, original_label, augmented_label, entities, existing_samples=None):
        reason = ""
        
        # 1. Label changes
        if original_label != augmented_label:
            return False, "Label changed"
            
        aug_lower = augmented_text.lower()
        orig_lower = original_text.lower()
        
        # 2. Entity changes
        for ent in entities:
            # Depending on how entities are tracked, we assume they are protected with [ENTITY_TYPE] or similar.
            # Alternatively, if they are plain strings:
            ent_str = ent.get('text', '').lower()
            if ent_str and ent_str not in aug_lower:
                return False, f"Entity '{ent_str}' lost"
                
        # 3. 'xxxx' token replaced
        orig_redactions = orig_lower.count('xxxx')
        aug_redactions = aug_lower.count('xxxx')
        if orig_redactions > 0 and aug_redactions < orig_redactions:
            return False, "'XXXX' token replaced"
            
        # 4. Hallucinated Banking Entity
        # Check if any banking entity is in the augmented text that was NOT in the original text
        for bank_ent in self.banking_entities:
            if bank_ent in aug_lower and bank_ent not in orig_lower:
                return False, f"Hallucinated Banking Entity: '{bank_ent}'"
                
        orig_tokens = set(orig_lower.split())
        aug_tokens = set(aug_lower.split())
        
        if not orig_tokens:
            return True, "Valid"
            
        overlap = len(orig_tokens.intersection(aug_tokens))
        similarity = overlap / len(orig_tokens)
        
        # 5. Semantic similarity below threshold
        if similarity < self.semantic_threshold:
            return False, f"Semantic similarity below threshold ({similarity:.2f} < {self.semantic_threshold})"
            
        # 6. Duplicate sample (exact match)
        if aug_lower == orig_lower:
            return False, "Duplicate sample (exact match with original)"
            
        if existing_samples:
            for sample in existing_samples:
                sample_lower = sample.lower()
                if aug_lower == sample_lower:
                    return False, "Duplicate sample (exact match with existing)"
                    
                sample_tokens = set(sample_lower.split())
                if len(sample_tokens) > 0:
                    samp_overlap = len(aug_tokens.intersection(sample_tokens))
                    jaccard = samp_overlap / (len(aug_tokens) + len(sample_tokens) - samp_overlap)
                    
                    # 7. Redundant sample
                    if jaccard > 0.90:
                        return False, "Redundant sample (Jaccard > 0.90)"
            
        return True, "Valid"

