import csv
import os
import datetime

class PolicyMemory:
    def __init__(self, capacity=1000, csv_path="results/policy_memory.csv"):
        self.capacity = capacity
        self.csv_path = csv_path
        self.memory = []
        self.alpha_wolves = []
        self.beta_wolves = []
        self.delta_wolves = []
        self.elite_population = []
        
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp", "Chunk_ID", "Macro_F1", "Precision", "Recall",
                    "BT_Ratio", "BERT_Ratio", "Budget", "Mask_Prob", "Sem_Thresh", 
                    "Ent_Weight", "Chunk_Pri", "LR"
                ])
                
    def add_state(self, chunk_id, policy, metrics, alpha, beta, delta, elite_pop):
        timestamp = datetime.datetime.now().isoformat()
        
        entry = {
            "chunk_id": chunk_id,
            "timestamp": timestamp,
            "policy": policy,
            "metrics": metrics,
            "alpha": alpha,
            "beta": beta,
            "delta": delta
        }
        self.memory.append(entry)
        
        self.alpha_wolves.append(alpha)
        self.beta_wolves.append(beta)
        self.delta_wolves.append(delta)
        self.elite_population = elite_pop
        
        if len(self.memory) > self.capacity:
            self.memory.pop(0)
            self.alpha_wolves.pop(0)
            self.beta_wolves.pop(0)
            self.delta_wolves.pop(0)
            
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, chunk_id, 
                metrics.get("macro_f1", 0), metrics.get("precision", 0), metrics.get("recall", 0),
                *policy
            ])
            
    def get_last_alpha_beta_delta(self):
        if self.alpha_wolves and self.beta_wolves and self.delta_wolves:
            return self.alpha_wolves[-1], self.beta_wolves[-1], self.delta_wolves[-1]
        return None, None, None
        
    def get_elite_population(self):
        return self.elite_population
