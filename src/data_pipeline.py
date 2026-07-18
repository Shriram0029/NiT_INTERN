import os
import json
import re
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
            "phishing", "scam", "wire fraud", "payment fraud"
        ]
        
        self.reject_keywords = [
            "mortgage", "home loan", "vehicle loan", "general credit disputes",
            "debt collection", "billing errors"
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
        
        available_columns = list(raw_data[0].keys()) if total_size > 0 else []
        
        labels = Counter()
        
        for item in raw_data:
            text = item.get("complaint_what_happened", "")
            labels[item.get("issue", "Unknown")] += 1
            
            if not text.strip():
                empty_count += 1
                continue
                
            if text in seen_narratives:
                duplicate_count += 1
            seen_narratives.add(text)
            
            words = text.split()
            total_words += len(words)
            
            # Count highly redacted
            redactions = sum(1 for w in words if 'xxxx' in w.lower())
            if len(words) > 0 and redactions / len(words) > 0.8:
                redacted_count += 1
                
        avg_length = total_words / max((total_size - empty_count), 1)
        empty_pct = (empty_count / max(total_size, 1)) * 100
        duplicate_pct = (duplicate_count / max(total_size, 1)) * 100
        redacted_pct = (redacted_count / max(total_size, 1)) * 100
        
        report = f"""# Data Profile

## Dataset Overview
- **Dataset Size**: {total_size}
- **Available Columns**: {', '.join(available_columns)}
- **Complaint Text Column**: complaint_what_happened
- **Label Column**: Derived from Product/Issue
- **Missing Values / Empty %**: {empty_pct:.2f}%
- **Duplicate %**: {duplicate_pct:.2f}%
- **Redaction % (Highly Redacted)**: {redacted_pct:.2f}%
- **Average Complaint Length (words)**: {avg_length:.2f}

## Label Distribution (Raw 'Issue' Column)
"""
        for lbl, cnt in labels.most_common(10):
            report += f"- {lbl}: {cnt}\n"
            
        report += "\n## Example Complaints\n"
        for i, item in enumerate([x for x in raw_data if x.get("complaint_what_happened", "").strip()][:3]):
            report += f"\n**Example {i+1}**:\n{item['complaint_what_happened'][:500]}...\n"

        with open(os.path.join(self.results_dir, "data_profile.md"), "w", encoding="utf-8") as f:
            f.write(report)
            
    def filter_cyber_banking(self, raw_data):
        filtered = []
        for item in raw_data:
            text = (item.get("complaint_what_happened", "") + " " + item.get("product", "") + " " + item.get("issue", "") + " " + item.get("sub_issue", "")).lower()
            
            has_cyber = any(kw in text for kw in self.cyber_keywords)
            has_reject = any(kw in text for kw in self.reject_keywords)
            
            if has_cyber or (not has_reject): # If it has reject words but is fraud related, we keep it.
                filtered.append(item)
        return filtered
        
    def clean_data(self, data):
        cleaned = []
        seen = set()
        
        for item in data:
            text = item.get("complaint_what_happened", "")
            
            # Lowercase & Whitespace
            text = text.lower()
            text = re.sub(r'\s+', ' ', text).strip()
            
            if not text:
                continue
                
            words = text.split()
            if len(words) < 5:
                continue
                
            # Full redaction check
            redactions = sum(1 for w in words if 'xxxx' in w)
            if redactions == len(words):
                continue
                
            if text in seen:
                continue
            seen.add(text)
            
            # Masking
            text = re.sub(r'\b\d{6}\b', '[OTP]', text)
            text = re.sub(r'\b(?:\d[ -]*?){13,16}\b', '[CARD]', text)
            
            item["complaint_what_happened_clean"] = text
            cleaned.append(item)
            
        return cleaned
        
    def assign_labels(self, data):
        labeled = []
        for item in data:
            text = (item.get("product", "") + " " + item.get("issue", "") + " " + item.get("sub_issue", "")).lower()
            
            assigned_label = "Unknown"
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
            train_val, test = train_test_split(data, test_size=0.1, stratify=labels, random_state=42)
            train_val_labels = [item["label"] for item in train_val]
            train, val = train_test_split(train_val, test_size=0.1111, stratify=train_val_labels, random_state=42) 
        except ValueError:
            train, test = train_test_split(data, test_size=0.1, random_state=42)
            train, val = train_test_split(train, test_size=0.1111, random_state=42)
            
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
