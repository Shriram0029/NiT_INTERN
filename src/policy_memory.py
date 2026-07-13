import csv
import os
import datetime
import numpy as np

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
                    "Timestamp", "Chunk_ID", "Fitness", "Macro_F1", "Semantic_Pres", "Entity_Pres",
                    "Cost", "BT_Ratio", "BERT_Ratio", "Budget", "Mask_Prob", "Sem_Thresh", 
                    "Ent_Weight", "Chunk_Pri", "LR", "Strategy"
                ])
                
    def add_state(self, chunk_id, policy, metrics, alpha, beta, delta, elite_pop, faci_dict=None):
        timestamp = datetime.datetime.now().isoformat()
        
        strategy = self._determine_strategy(policy[0], policy[1])
        
        entry = {
            "chunk_id": chunk_id,
            "timestamp": timestamp,
            "policy": policy,
            "strategy": strategy,
            "metrics": metrics,
            "alpha": alpha,
            "beta": beta,
            "delta": delta,
            "elite_pop": elite_pop,
            "faci_dict": faci_dict
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
                metrics.get("fitness", 0), metrics.get("macro_f1", 0), 
                metrics.get("semantic_preservation", 0), metrics.get("entity_preservation", 0),
                metrics.get("cost", 0),
                *policy, strategy
            ])
            
    def _determine_strategy(self, p_bt, p_bert):
        if p_bt == 0 and p_bert == 0:
            return "No Augmentation"
        elif p_bt > 0.5 and p_bert > 0.5:
            return "Hybrid"
        elif p_bt > p_bert:
            return "Back Translation"
        else:
            return "BERT Contextual"
            
    def get_last_alpha_beta_delta(self):
        if self.alpha_wolves and self.beta_wolves and self.delta_wolves:
            return self.alpha_wolves[-1], self.beta_wolves[-1], self.delta_wolves[-1]
        return None, None, None
        
    def get_elite_population(self):
        return self.elite_population

    def get_elites_by_faci_similarity(self, current_faci):
        if not self.memory or not current_faci:
            return self.elite_population
            
        def _faci_vector(f_dict):
            if not f_dict:
                return np.zeros(10)
            return np.array([
                f_dict.get('complexity', 0),
                f_dict.get('entity_density', 0),
                f_dict.get('fraud_density', 0),
                f_dict.get('risk_score', 0),
                f_dict.get('ambiguity', 0),
                f_dict.get('semantic_entropy', 0),
                f_dict.get('rare_word_density', 0),
                f_dict.get('redaction_density', 0),
                f_dict.get('class_imbalance_factor', 1.0),
                f_dict.get('recommended_budget', 1)
            ])
            
        curr_vec = _faci_vector(current_faci)
        norm_curr = np.linalg.norm(curr_vec)
        if norm_curr == 0:
            norm_curr = 1e-9
            
        best_sim = -float('inf')
        best_elites = self.elite_population
        
        for entry in self.memory:
            if 'faci_dict' in entry and entry['faci_dict']:
                hist_vec = _faci_vector(entry['faci_dict'])
                norm_hist = np.linalg.norm(hist_vec)
                if norm_hist == 0:
                    norm_hist = 1e-9
                    
                # Cosine Similarity
                sim = np.dot(curr_vec, hist_vec) / (norm_curr * norm_hist)
                
                if sim > best_sim:
                    best_sim = sim
                    if 'elite_pop' in entry:
                        best_elites = entry['elite_pop']
                    
        return best_elites

