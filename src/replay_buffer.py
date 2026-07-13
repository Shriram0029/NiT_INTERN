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
        self.total_seen = 0
        
    def add(self, samples):
        """
        samples: list of tuples (text, label)
        Uses Reservoir Sampling
        """
        for sample in samples:
            self.total_seen += 1
            text, label = sample
            
            if len(self.buffer) < self.capacity:
                self.buffer.append(sample)
                self.class_indices[label].append(len(self.buffer) - 1)
            else:
                j = random.randint(0, self.total_seen - 1)
                if j < self.capacity:
                    old_label = self.buffer[j][1]
                    self.class_indices[old_label].remove(j)
                    
                    self.buffer[j] = sample
                    self.class_indices[label].append(j)
                    
    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return []
            
        if len(self.buffer) < batch_size:
            return self.buffer.copy()
            
        # Balanced batch sampling
        samples_per_class = max(1, batch_size // max(1, len(self.class_indices)))
        
        batch_indices = []
        classes_available = list(self.class_indices.keys())
        
        for c in classes_available:
            indices_for_c = self.class_indices[c]
            if not indices_for_c:
                continue
                
            num_to_sample = min(samples_per_class, len(indices_for_c))
            batch_indices.extend(random.sample(indices_for_c, num_to_sample))
            
        # Fill remaining if needed
        remaining = batch_size - len(batch_indices)
        if remaining > 0:
            all_other_indices = list(set(range(len(self.buffer))) - set(batch_indices))
            if all_other_indices:
                num_to_fill = min(remaining, len(all_other_indices))
                batch_indices.extend(random.sample(all_other_indices, num_to_fill))
                
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

