import os
import csv
import datetime

class SemanticValidator:
    def __init__(self, semantic_threshold=0.8, entity_weight=1.0, report_path="results/validation_report.csv"):
        self.semantic_threshold = semantic_threshold
        self.entity_weight = entity_weight
        self.report_path = report_path
        
        self.banking_entities = ["bank", "credit card", "account", "loan", "mortgage", "debit", "transaction", "payment", "fund", "wire", "transfer"]
        
        os.makedirs(os.path.dirname(self.report_path), exist_ok=True)
        if not os.path.exists(self.report_path):
            with open(self.report_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Chunk_ID", "Original_Text", "Augmented_Text", "Status", "Reason"])
        
    def validate_and_log(self, chunk_id, original_text, augmented_text, original_label, augmented_label, entities, existing_samples=None):
        is_valid, reason = self._validate(original_text, augmented_text, original_label, augmented_label, entities, existing_samples)
        
        timestamp = datetime.datetime.now().isoformat()
        with open(self.report_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, chunk_id, original_text, augmented_text, "Accepted" if is_valid else "Rejected", reason])
            
        return is_valid, reason

    def _validate(self, original_text, augmented_text, original_label, augmented_label, entities, existing_samples=None):
        if not augmented_text or not augmented_text.strip():
            return False, "Empty generation"
            
        if original_label != augmented_label:
            return False, "Label changed"
            
        aug_lower = augmented_text.lower()
        orig_lower = original_text.lower()
        
        for ent in entities:
            # entities could be "[OTP]", "[CARD]", etc since we masked them
            ent_str = ent.lower()
            if ent_str and ent_str not in aug_lower:
                return False, f"Entity '{ent_str}' lost"
                
        orig_redactions = orig_lower.count('xxxx')
        aug_redactions = aug_lower.count('xxxx')
        if orig_redactions > 0 and aug_redactions < orig_redactions:
            return False, "'XXXX' token replaced"
            
        for bank_ent in self.banking_entities:
            if bank_ent in aug_lower and bank_ent not in orig_lower:
                return False, f"Hallucinated Banking Entity: '{bank_ent}'"
                
        orig_tokens = set(orig_lower.split())
        aug_tokens = set(aug_lower.split())
        
        if not orig_tokens:
            return True, "Valid"
            
        overlap = len(orig_tokens.intersection(aug_tokens))
        similarity = overlap / len(orig_tokens)
        
        if similarity < self.semantic_threshold:
            return False, f"Semantic similarity below threshold ({similarity:.2f} < {self.semantic_threshold})"
            
        if aug_lower == orig_lower:
            return False, "Duplicate sample (exact match with original)"
            
        if existing_samples:
            for sample in existing_samples:
                if aug_lower == sample.lower():
                    return False, "Duplicate sample (exact match with existing)"
            
        return True, "Valid"
