import os
import json
import re
import unicodedata
from collections import Counter
from sklearn.model_selection import train_test_split

class DataPipeline:
    def __init__(self, data_path, results_dir="results"):
        self.data_path = data_path
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.cyber_keywords = [
            "unauthorized transaction", "identity theft", "card fraud",
            "account takeover", "online banking fraud", "upi fraud",
            "payment fraud", "phishing", "scam", "wire transfer fraud", "fraud"
        ]
        
        self.reject_keywords = [
            "mortgage", "vehicle loan", "student loan", "debt collection", "billing errors"
        ]
        
        self.supported_classes = [
            "Unauthorized Transaction",
            "Identity Theft",
            "Card Fraud",
            "Account Takeover",
            "Phishing / Scam"
        ]
        
    def run_pipeline(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            
        self.analyze_dataset(raw_data)
        
        filtered_data = self.filter_cyber_banking(raw_data)
        cleaned_data = self.clean_data(filtered_data)
        labeled_data = self.assign_labels(cleaned_data)
        
        # Save filtered dataset
        filtered_path = os.path.join(os.path.dirname(self.data_path), "filtered_dataset.json")
        with open(filtered_path, "w", encoding="utf-8") as f:
            json.dump(labeled_data, f, indent=4)
            
        self.stratified_split(labeled_data)
        
    def analyze_dataset(self, raw_data):
        total_size = len(raw_data)
        empty_count = 0
        duplicate_count = 0
        redacted_count = 0
        total_words = 0
        seen_narratives = set()
        vocab = set()
        
        labels = Counter()
        
        for item in raw_data:
            text = item.get("complaint_what_happened", "")
            # Assuming 'label' might not exist, but let's check
            label = item.get("label", item.get("issue", "Unknown"))
            labels[label] += 1
            
            if not text or not text.strip():
                empty_count += 1
                continue
                
            if text in seen_narratives:
                duplicate_count += 1
            seen_narratives.add(text)
            
            words = text.split()
            total_words += len(words)
            vocab.update(w.lower() for w in words)
            
            # Count highly redacted (fully redacted)
            redactions = sum(1 for w in words if 'xxxx' in w.lower())
            if len(words) > 0 and redactions == len(words):
                redacted_count += 1
                
        avg_length = total_words / max((total_size - empty_count), 1)
        empty_pct = (empty_count / max(total_size, 1)) * 100
        duplicate_pct = (duplicate_count / max(total_size, 1)) * 100
        redacted_pct = (redacted_count / max(total_size, 1)) * 100
        vocab_size = len(vocab)
        
        report = f"""# Data Profile

## Dataset Overview
- **Dataset Size**: {total_size}
- **Complaint Column Detected**: complaint_what_happened
- **Product Column Detected**: product
- **Issue Column Detected**: issue
- **Sub Issue Column Detected**: sub_issue
- **Missing %**: {empty_pct:.2f}%
- **Duplicate %**: {duplicate_pct:.2f}%
- **Redacted %**: {redacted_pct:.2f}%
- **Average Complaint Length (words)**: {avg_length:.2f}
- **Vocabulary Size**: {vocab_size}

## Class Distribution (Based on raw issue/label)
"""
        for lbl, cnt in labels.most_common(20):
            report += f"- {lbl}: {cnt}\n"
            
        with open(os.path.join(self.results_dir, "data_profile.md"), "w", encoding="utf-8") as f:
            f.write(report)
            
    def filter_cyber_banking(self, raw_data):
        filtered = []
        for item in raw_data:
            text = (item.get("complaint_what_happened", "") + " " + item.get("product", "") + " " + item.get("issue", "") + " " + item.get("sub_issue", "")).lower()
            
            has_cyber = any(kw in text for kw in self.cyber_keywords)
            has_reject = any(kw in text for kw in self.reject_keywords)
            
            if has_cyber or (not has_reject): 
                filtered.append(item)
        return filtered
        
    def clean_data(self, data):
        cleaned = []
        seen = set()
        
        for item in data:
            text = item.get("complaint_what_happened", "")
            
            if not text or not text.strip():
                continue
                
            # Normalize: Lowercase, Unicode, Whitespace
            text = str(text).lower()
            text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
            text = re.sub(r'\s+', ' ', text).strip()
            
            words = text.split()
            if len(words) < 5:
                continue
                
            # Fully redacted check
            redactions = sum(1 for w in words if 'xxxx' in w)
            if len(words) > 0 and redactions == len(words):
                continue
                
            if text in seen:
                continue
            seen.add(text)
            
            # Masking Patterns
            text = re.sub(r'\b\d{4,6}\b', '[OTP]', text) # Simplified OTP masking
            text = re.sub(r'\b(?:\d[ -]*?){13,16}\b', '[CARD]', text) # Card masking
            text = re.sub(r'\b\d{9,18}\b', '[ACCOUNT]', text) # Account number
            text = re.sub(r'\b[a-zA-Z0-9.\-_]{2,256}@[a-zA-Z]{2,64}\.[a-zA-Z]{2,10}\b', '[UPI]', text) # UPI / Email
            text = re.sub(r'\b[A-Z]{4}0[A-Z0-9]{6}\b', '[IFSC]', text) # IFSC code
            
            item["complaint_what_happened_clean"] = text
            cleaned.append(item)
            
        return cleaned
        
    def assign_labels(self, data):
        labeled = []
        for item in data:
            if "label" in item and item["label"] in self.supported_classes:
                # Label already exists and is supported
                labeled.append(item)
                continue
                
            text = (item.get("product", "") + " " + item.get("issue", "") + " " + item.get("sub_issue", "")).lower()
            
            assigned_label = "Phishing / Scam" # Default fallback for cyber-filtered data
            if "unauthorized" in text or "recognize" in text:
                assigned_label = "Unauthorized Transaction"
            elif "identity" in text or "belong" in text:
                assigned_label = "Identity Theft"
            elif "card" in text:
                assigned_label = "Card Fraud"
            elif "takeover" in text or "login" in text:
                assigned_label = "Account Takeover"
            elif "scam" in text or "phishing" in text or "fraud" in text:
                assigned_label = "Phishing / Scam"
            else:
                comp_text = item.get("complaint_what_happened_clean", "").lower()
                if "unauthorized" in comp_text: assigned_label = "Unauthorized Transaction"
                elif "identity" in comp_text: assigned_label = "Identity Theft"
                elif "card" in comp_text: assigned_label = "Card Fraud"
                elif "takeover" in comp_text: assigned_label = "Account Takeover"
                elif "scam" in comp_text or "fraud" in comp_text: assigned_label = "Phishing / Scam"
                else: assigned_label = "Unauthorized Transaction"
                
            item["label"] = assigned_label
            labeled.append(item)
            
        label_mapping = {label: idx for idx, label in enumerate(self.supported_classes)}
        with open(os.path.join(self.results_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
            json.dump(label_mapping, f, indent=4)
            
        return labeled
        
    def stratified_split(self, data):
        labels = [item["label"] for item in data]
        
        try:
            # Train (80%), Val (10%), Test (10%)
            train_val, test = train_test_split(data, test_size=0.10, stratify=labels, random_state=42)
            train_val_labels = [item["label"] for item in train_val]
            # Validation is 1/9 of train_val to get 10% of total
            train, val = train_test_split(train_val, test_size=1/9, stratify=train_val_labels, random_state=42) 
        except ValueError:
            # Fallback if classes are too small for stratified split
            train_val, test = train_test_split(data, test_size=0.10, random_state=42)
            train, val = train_test_split(train_val, test_size=1/9, random_state=42)
            
        data_dir = os.path.dirname(self.data_path)
        with open(os.path.join(data_dir, "train.json"), "w", encoding="utf-8") as f:
            json.dump(train, f, indent=4)
        with open(os.path.join(data_dir, "validation.json"), "w", encoding="utf-8") as f:
            json.dump(val, f, indent=4)
        with open(os.path.join(data_dir, "test.json"), "w", encoding="utf-8") as f:
            json.dump(test, f, indent=4)
            
if __name__ == "__main__":
    pipeline = DataPipeline("data/complaints_150.json")
    pipeline.run_pipeline()
