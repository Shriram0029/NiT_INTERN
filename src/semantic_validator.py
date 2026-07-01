class SemanticValidator:
    def __init__(self, semantic_threshold=0.8, entity_weight=1.0):
        self.semantic_threshold = semantic_threshold
        self.entity_weight = entity_weight
        
    def validate(self, original_text, augmented_text, entities):
        for ent in entities:
            if ent['type'] in ['OTP', 'CARD']:
                placeholder = f"[{ent['type']}]"
                if placeholder not in augmented_text:
                    return False
                    
        orig_tokens = set(original_text.lower().split())
        aug_tokens = set(augmented_text.lower().split())
        
        if not orig_tokens:
            return True
            
        overlap = len(orig_tokens.intersection(aug_tokens))
        similarity = overlap / len(orig_tokens)
        
        if similarity < self.semantic_threshold:
            return False
            
        return True
