import random

class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.buffer = []
        
    def add(self, samples):
        """
        samples: list of tuples (text, label)
        """
        self.buffer.extend(samples)
        if len(self.buffer) > self.capacity:
            self.buffer = random.sample(self.buffer, self.capacity)
            
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return self.buffer
        return random.sample(self.buffer, batch_size)
