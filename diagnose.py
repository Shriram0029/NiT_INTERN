"""
Diagnostic script: runs 3 chunks without augmentation to isolate crash
"""
import os, sys, warnings, gc, time
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'
warnings.filterwarnings('ignore')

import torch
torch.set_num_threads(1)
import json, numpy as np

with open('data/train.json') as f: train_data = json.load(f)
with open('data/test.json') as f: test_data = json.load(f)
with open('results/label_mapping.json') as f: label_mapping = json.load(f)
classes = list(label_mapping.keys())
num_classes = len(classes)

from src.faci import FACICalculator
from src.replay_buffer import ReplayBuffer
from src.train_classifier import IncrementalClassifier
from src.evaluate import Evaluator

faci = FACICalculator(results_dir='results')
classifier = IncrementalClassifier(num_classes=num_classes, checkpoint_dir='results/checkpoints')
evaluator = Evaluator(num_classes=num_classes)
rb = ReplayBuffer(capacity=1000, num_classes=num_classes)
test_batch = [(item['complaint_what_happened_clean'], label_mapping[item['label']]) for item in test_data]

chunk_size = 10
for chunk_id in range(4):
    chunk = train_data[chunk_id * chunk_size:(chunk_id + 1) * chunk_size]
    t0 = time.time()
    faci_vecs = [faci.compute(item['complaint_what_happened_clean']) for item in chunk]
    avg_scalar = float(np.mean([f['scalar'] for f in faci_vecs]))

    batch = [(item['complaint_what_happened_clean'], label_mapping[item['label']]) for item in chunk]
    rb.add(batch)
    train_batch = rb.sample(batch_size=16)
    loss = classifier.train_on_batch(train_batch, learning_rate=2e-5)
    eval_res = evaluator.evaluate(classifier.model, classifier.tokenizer, test_batch, classifier.device)
    f1 = eval_res.get('macro_f1', 0.0)
    elapsed = time.time() - t0
    print(f"Chunk {chunk_id} | Loss: {loss:.3f} | F1: {f1:.3f} | Time: {elapsed:.1f}s")
    gc.collect()

print("DIAGNOSTIC PASSED")
