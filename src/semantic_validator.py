import os
import csv
import datetime

class SemanticValidator:
    def __init__(self, semantic_threshold=0.8, entity_weight=1.0, report_path="results/validation_report.csv"):
        self.semantic_threshold = semantic_threshold
        self.semantic_upper_bound = 0.98
        self.entity_weight = entity_weight
        self.report_path = report_path
        
        self.banking_entities = ["bank", "credit card", "account", "loan", "mortgage", "debit", "transaction", "payment", "fund", "wire", "transfer"]
        
        os.makedirs(os.path.dirname(self.report_path), exist_ok=True)
        if not os.path.exists(self.report_path):
            with open(self.report_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Chunk_ID", "Original_Text", "Augmented_Text", "Similarity", "Threshold", "Method", "Status", "Reason"])
        
    def validate_and_log(self, chunk_id, original_text, augmented_text, original_label, augmented_label, entities, existing_samples=None, method="Unknown", dynamic_threshold=None):
        threshold_to_use = dynamic_threshold if dynamic_threshold is not None else self.semantic_threshold
        is_valid, reason, similarity = self._validate(original_text, augmented_text, original_label, augmented_label, entities, existing_samples, threshold_to_use)
        
        timestamp = datetime.datetime.now().isoformat()
        with open(self.report_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, chunk_id, original_text, augmented_text, 
                f"{similarity:.4f}" if similarity is not None else "N/A",
                f"{threshold_to_use:.2f}",
                method, 
                "Accepted" if is_valid else "Rejected", 
                reason
            ])
            
        return is_valid, reason

    def _validate(self, original_text, augmented_text, original_label, augmented_label, entities, existing_samples=None, threshold=0.8):
        if not augmented_text or not augmented_text.strip():
            return False, "Empty generation", 0.0
            
        if original_label != augmented_label:
            return False, "Label changed", 0.0
            
        aug_lower = augmented_text.lower()
        orig_lower = original_text.lower()
        
        if aug_lower == orig_lower:
            return False, "Exact Duplicate", 1.0
            
        for ent in entities:
            ent_str = ent.lower()
            if ent_str and ent_str not in aug_lower:
                return False, f"Entity Change: '{ent_str}' lost", None
                
        orig_redactions = orig_lower.count('xxxx')
        aug_redactions = aug_lower.count('xxxx')
        if orig_redactions > 0 and aug_redactions < orig_redactions:
            return False, "XXXX replaced", None
            
        for bank_ent in self.banking_entities:
            orig_words = set(orig_lower.split())
            aug_words = set(aug_lower.split())
            if bank_ent in aug_words and bank_ent not in orig_words:
                return False, f"Hallucinated Banking Entity: '{bank_ent}'", None
                
        orig_tokens = set(orig_lower.split())
        aug_tokens = set(aug_lower.split())
        
        if not orig_tokens:
            return True, "Valid", 1.0
            
        overlap = len(orig_tokens.intersection(aug_tokens))
        similarity = overlap / len(orig_tokens)
        
        if similarity < threshold:
            return False, f"Semantic similarity below threshold ({similarity:.2f} < {threshold})", similarity
            
        if similarity > self.semantic_upper_bound:
            return False, f"Semantic similarity too high ({similarity:.2f} > {self.semantic_upper_bound})", similarity
            
        if existing_samples:
            for sample in existing_samples:
                if aug_lower == sample.lower():
                    return False, "Duplicate sample (exact match with existing)", similarity
            
        return True, "Valid", similarity
