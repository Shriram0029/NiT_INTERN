import csv
import os
import datetime
import numpy as np

class PolicyMemory:
    def __init__(self, capacity=1000, csv_path="results/policy_memory.csv"):
        self.capacity = capacity
        self.csv_path = csv_path
        self.memory = []
        
        # GWO specific tracking
        self.alpha_wolves = []
        self.beta_wolves = []
        self.delta_wolves = []
        self.elite_population = []
        self.seen_alphas = set() # For deduplication
        
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp", "Chunk_ID", "Strategy", "Budget", "Fitness", "Macro_F1", 
                    "Utility", "Confidence", "Cost", "FACI_Vector", "Semantic_Threshold"
                ])
                
    def add_state(self, chunk_id, faci_vector, strategy, fitness, macro_f1, utility, confidence, cost, budget=0, alpha=None, beta=None, delta=None, elite_pop=None, semantic_threshold=None):
        timestamp = datetime.datetime.now().isoformat()
        
        # Deduplication check
        if alpha is not None:
            alpha_tuple = tuple(np.round(alpha, 4))
            if alpha_tuple in self.seen_alphas:
                pass 
            else:
                self.seen_alphas.add(alpha_tuple)
        
        entry = {
            "chunk_id": chunk_id,
            "timestamp": timestamp,
            "faci_vector": faci_vector,
            "strategy": strategy,
            "budget": budget,
            "fitness": fitness,
            "macro_f1": macro_f1,
            "utility": utility,
            "confidence": confidence,
            "cost": cost,
            "alpha": alpha,
            "beta": beta,
            "delta": delta,
            "elite_pop": elite_pop or [],
            "semantic_threshold": semantic_threshold
        }
        
        self.memory.append(entry)
        
        if alpha is not None: self.alpha_wolves.append(alpha)
        if beta is not None: self.beta_wolves.append(beta)
        if delta is not None: self.delta_wolves.append(delta)
        if elite_pop is not None and len(elite_pop) > 0: 
            # Deduplicate elites before assigning
            unique_elites = []
            seen_elites = set()
            for e in elite_pop:
                t = tuple(np.round(e, 4))
                if t not in seen_elites:
                    seen_elites.add(t)
                    unique_elites.append(e)
            self.elite_population = unique_elites

        if len(self.memory) > self.capacity:
            removed = self.memory.pop(0)
            if removed.get("alpha") is not None:
                a_t = tuple(np.round(removed["alpha"], 4))
                if a_t in self.seen_alphas:
                    self.seen_alphas.remove(a_t)
            if self.alpha_wolves: self.alpha_wolves.pop(0)
            if self.beta_wolves: self.beta_wolves.pop(0)
            if self.delta_wolves: self.delta_wolves.pop(0)
            
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, chunk_id, strategy, budget, fitness, macro_f1, 
                utility, confidence, cost, str(faci_vector), semantic_threshold
            ])
            
    def get_last_alpha_beta_delta(self):
        alpha = self.alpha_wolves[-1] if self.alpha_wolves else None
        beta = self.beta_wolves[-1] if self.beta_wolves else None
        delta = self.delta_wolves[-1] if self.delta_wolves else None
        return alpha, beta, delta
        
    def get_elite_population(self):
        return self.elite_population

    def get_elites_by_faci_similarity(self, current_faci_vector):
        """
        Retrieve previous elites based on Cosine Similarity of FACI vectors.
        """
        if not self.memory or current_faci_vector is None:
            return self.elite_population
            
        curr_vec = np.array(current_faci_vector)
        norm_curr = np.linalg.norm(curr_vec)
        if norm_curr == 0:
            norm_curr = 1e-9
            
        best_sim = -float('inf')
        best_elites = self.elite_population
        
        for entry in self.memory:
            if 'faci_vector' in entry and entry['faci_vector'] is not None:
                hist_vec = np.array(entry['faci_vector'])
                norm_hist = np.linalg.norm(hist_vec)
                if norm_hist == 0:
                    norm_hist = 1e-9
                    
                # Cosine Similarity
                sim = np.dot(curr_vec, hist_vec) / (norm_curr * norm_hist)
                
                if sim > best_sim:
                    best_sim = sim
                    if 'elite_pop' in entry and entry['elite_pop']:
                        best_elites = entry['elite_pop']
                        
        return best_elites
