import json
import random
from collections import Counter
import re
import os

def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'xxxx+', 'xxxx', text)
    text = re.sub(r'\{\$\d+\.\d+\}', '{$[OTP].00}', text)
    return text.strip()

def main():
    print("Loading data1.json...")
    with open('data/data1.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Total raw records: {len(data)}")
    
    # Filter non-empty complaints
    valid_data = [d for d in data if d.get('complaint_what_happened', '').strip()]
    print(f"Records with text: {len(valid_data)}")
    
    # Get top 5 classes by 'product'
    product_counts = Counter(d['product'] for d in valid_data)
    top_5 = [item[0] for item in product_counts.most_common(5)]
    print(f"Top 5 classes: {top_5}")
    
    # Create mapping
    label_mapping = {label: i for i, label in enumerate(top_5)}
    
    # Filter dataset to only those top 5 classes
    filtered_data = []
    for d in valid_data:
        if d['product'] in top_5:
            d['label'] = d['product']
            d['complaint_what_happened_clean'] = clean_text(d['complaint_what_happened'])
            filtered_data.append(d)
            
    print(f"Records in top 5 classes: {len(filtered_data)}")
    
    # Shuffle and split 80/20
    random.seed(42)
    random.shuffle(filtered_data)
    split_idx = int(len(filtered_data) * 0.8)
    
    train_data = filtered_data[:split_idx]
    test_data = filtered_data[split_idx:]
    
    print(f"Train size: {len(train_data)}")
    print(f"Test size: {len(test_data)}")
    
    # Write files
    os.makedirs('results', exist_ok=True)
    with open('results/label_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, indent=4)
        
    with open('data/train.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=4)
        
    with open('data/test.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=4)
        
    print("Data preparation complete!")

if __name__ == '__main__':
    main()
