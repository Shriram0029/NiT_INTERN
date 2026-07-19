import random
from collections import defaultdict
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=1000, num_classes=5):
        self.capacity = capacity
        self.num_classes = num_classes
        self.buffer = []
        
        # Keep track of indices per class
        self.class_indices = defaultdict(list)
        self.total_seen_per_class = defaultdict(int)
        self.total_seen = 0
        
    def add(self, samples):
        """
        samples: list of tuples (text, label)
        Uses Balanced Reservoir Sampling
        """
        for sample in samples:
            text, label = sample
            self.total_seen += 1
            self.total_seen_per_class[label] += 1
            
            # Per-class reservoir capacity (approx equal division)
            class_capacity = self.capacity // self.num_classes
            
            if len(self.class_indices[label]) < class_capacity:
                self.buffer.append(sample)
                self.class_indices[label].append(len(self.buffer) - 1)
            else:
                j = random.randint(0, self.total_seen_per_class[label] - 1)
                if j < class_capacity:
                    # Find the actual buffer index for this class's j-th element
                    buffer_idx = self.class_indices[label][j]
                    self.buffer[buffer_idx] = sample
                    
    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return []
            
        if len(self.buffer) < batch_size:
            return self.buffer.copy()
            
        # Balanced batch sampling with max 35% per class rule
        max_samples_per_class = int(batch_size * 0.35)
        samples_per_class = max(1, batch_size // self.num_classes)
        
        batch_indices = []
        classes_available = list(self.class_indices.keys())
        
        for c in classes_available:
            indices_for_c = self.class_indices[c]
            if not indices_for_c:
                continue
                
            num_to_sample = min(samples_per_class, len(indices_for_c), max_samples_per_class)
            # Weighted random sampling inside the class (could use weights if needed, here uniform over reservoir)
            batch_indices.extend(random.sample(indices_for_c, num_to_sample))
            
        # Fill remaining if needed
        remaining = batch_size - len(batch_indices)
        if remaining > 0:
            all_other_indices = list(set(range(len(self.buffer))) - set(batch_indices))
            if all_other_indices:
                # To still respect the 35% limit, we need to carefully fill
                random.shuffle(all_other_indices)
                class_counts = defaultdict(int)
                for idx in batch_indices:
                    class_counts[self.buffer[idx][1]] += 1
                    
                for idx in all_other_indices:
                    if remaining <= 0: break
                    lbl = self.buffer[idx][1]
                    if class_counts[lbl] < max_samples_per_class:
                        batch_indices.append(idx)
                        class_counts[lbl] += 1
                        remaining -= 1
                
                # If still remaining (e.g. extremely few classes exist), force fill regardless of 35% rule
                if remaining > 0:
                    still_available = list(set(range(len(self.buffer))) - set(batch_indices))
                    num_to_fill = min(remaining, len(still_available))
                    if num_to_fill > 0:
                        batch_indices.extend(random.sample(still_available, num_to_fill))
                
        # Shuffle batch
        random.shuffle(batch_indices)
        
        return [self.buffer[i] for i in batch_indices]
        
    def get_statistics(self):
        return {
            "total_capacity": self.capacity,
            "current_size": len(self.buffer),
            "class_distribution": {k: len(v) for k, v in self.class_indices.items()},
            "total_seen": self.total_seen
        }
