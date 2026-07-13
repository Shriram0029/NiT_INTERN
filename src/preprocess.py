import re
import os
import json
from collections import Counter
from src.entity_extractor import EntityExtractor

class Preprocessor:
    def __init__(self, config):
        self.extractor = EntityExtractor(config.get('protected_entities', []))
        
        # Cyber Banking Keywords
        self.cyber_keywords = [
            "identity theft", "unauthorized transaction", "debit card fraud", 
            "credit card fraud", "account takeover", "online banking", 
            "digital payment", "phishing", "scam", "wire transfer fraud", 
            "payment fraud", "otp", "upi", "authentication", "fraud", "stolen"
        ]
        
        self.reject_keywords = [
            "mortgage", "loan", "credit repair", "debt collection", "vehicle loan"
        ]

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def is_admissible(self, text, min_tokens=5, max_redaction_ratio=0.5):
        tokens = text.split()
        if len(tokens) < min_tokens:
            return False
        
        redaction_count = sum(1 for t in tokens if 'xxxx' in t.lower())
        if (redaction_count / max(len(tokens), 1)) > max_redaction_ratio:
            return False
            
        return True
        
    def is_cyber_banking_related(self, text, product=""):
        combined_text = (text + " " + product).lower()
        
        has_cyber = any(kw in combined_text for kw in self.cyber_keywords)
        has_reject = any(kw in combined_text for kw in self.reject_keywords)
        
        if has_reject and not has_cyber:
            return False
        if has_cyber:
            return True
        return False

    def process(self, complaint_text):
        if not self.is_admissible(complaint_text):
            return None
            
        clean = self.clean_text(complaint_text)
        processed_text, entities = self.extractor.extract(clean)
        return {
            "original": complaint_text,
            "processed_text": processed_text,
            "entities": entities
        }

    def filter_and_audit(self, raw_data, output_dir):
        """
        Filters raw dataset and generates data_audit_report.md
        raw_data: list of dicts {"product": ..., "complaint_what_happened": ...}
        """
        total_records = len(raw_data)
        filtered_records = []
        rejected_count = 0
        empty_count = 0
        total_redactions = 0
        total_tokens = 0
        
        products = []
        
        # Ensure distinct by tracking seen texts to compute duplicate %
        seen_texts = set()
        duplicate_count = 0

        for item in raw_data:
            text = item.get("complaint_what_happened", "").strip()
            product = item.get("product", "")
            
            if not text:
                empty_count += 1
                continue
                
            if text in seen_texts:
                duplicate_count += 1
                # Still process it, or reject? The prompt says "Duplicate %" in report.
            else:
                seen_texts.add(text)
                
            if not self.is_cyber_banking_related(text, product):
                rejected_count += 1
                continue
                
            tokens = text.split()
            total_tokens += len(tokens)
            total_redactions += sum(1 for t in tokens if 'xxxx' in t.lower())
            
            filtered_records.append(item)
            products.append(product)
            
        filtered_count = len(filtered_records)
        class_distribution = Counter(products)
        
        duplicate_percent = (duplicate_count / total_records) * 100 if total_records > 0 else 0
        empty_percent = (empty_count / total_records) * 100 if total_records > 0 else 0
        avg_length = total_tokens / filtered_count if filtered_count > 0 else 0
        redaction_percent = (total_redactions / total_tokens) * 100 if total_tokens > 0 else 0
        
        report_content = f"""# Data Audit Report

## Dataset Statistics
- **Total Records**: {total_records}
- **Filtered Records**: {filtered_count}
- **Rejected Records**: {rejected_count}
- **Empty Complaint %**: {empty_percent:.2f}%
- **Duplicate %**: {duplicate_percent:.2f}%
- **Average Length (tokens)**: {avg_length:.2f}
- **Redaction %**: {redaction_percent:.2f}%

## Class Distribution (Filtered)
"""
        for cls, count in class_distribution.items():
            report_content += f"- **{cls}**: {count}\n"
            
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "data_audit_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
            
        return filtered_records

