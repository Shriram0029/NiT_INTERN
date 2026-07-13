import numpy as np
import random

class GWOOptimizer:
    def __init__(self, config, bounds, memory=None):
        self.num_wolves = config['num_wolves']
        self.max_iter = config['max_iter']
        self.a_start = config['a_decay_start']
        self.a_end = config['a_decay_end']
        self.bounds = bounds
        self.num_dim = len(bounds)
        self.memory = memory
        
        self.fitness_history = []
        self.alpha_fitness_history = []
        self.beta_fitness_history = []
        self.delta_fitness_history = []
        self.wolf_distance_history = []
        self.search_radius_history = []
        
    def _clip(self, pos):
        for i in range(self.num_dim):
            pos[i] = np.clip(pos[i], self.bounds[i][0], self.bounds[i][1])
        return pos
        
    def optimize(self, initial_population, fitness_function):
        positions = []
        
        # Stateful Initialization
        if self.memory:
            p_alpha, p_beta, p_delta = self.memory.get_last_alpha_beta_delta()
            if p_alpha is not None: positions.append(p_alpha)
            if p_beta is not None: positions.append(p_beta)
            if p_delta is not None: positions.append(p_delta)
            
        for ind in initial_population:
            positions.append(ind)
            
        positions = np.array(positions)
        
        while len(positions) < self.num_wolves:
            ind = [random.uniform(b[0], b[1]) for b in self.bounds]
            positions = np.vstack([positions, ind])
            
        if len(positions) > self.num_wolves:
            positions = positions[:self.num_wolves]
            
        alpha_pos, alpha_score = None, float('-inf')
        beta_pos, beta_score = None, float('-inf')
        delta_pos, delta_score = None, float('-inf')
        
        for t in range(self.max_iter):
            for i in range(self.num_wolves):
                positions[i] = self._clip(positions[i])
                score = fitness_function(positions[i])
                
                if score > alpha_score:
                    delta_score, delta_pos = beta_score, beta_pos
                    beta_score, beta_pos = alpha_score, alpha_pos
                    alpha_score, alpha_pos = score, positions[i].copy()
                elif score > beta_score:
                    delta_score, delta_pos = beta_score, beta_pos
                    beta_score, beta_pos = score, positions[i].copy()
                elif score > delta_score:
                    delta_score, delta_pos = score, positions[i].copy()
                    
            self.fitness_history.append(alpha_score)
            self.alpha_fitness_history.append(alpha_score)
            self.beta_fitness_history.append(beta_score)
            self.delta_fitness_history.append(delta_score)
            
            # Calculate mean wolf distance
            mean_pos = np.mean(positions, axis=0)
            distances = [np.linalg.norm(pos - mean_pos) for pos in positions]
            self.wolf_distance_history.append(np.mean(distances))
            
            a = self.a_start - t * ((self.a_start - self.a_end) / self.max_iter)
            self.search_radius_history.append(a)
            
            for i in range(self.num_wolves):
                for j in range(self.num_dim):
                    r1, r2 = random.random(), random.random()
                    A1, C1 = 2 * a * r1 - a, 2 * r2
                    D_alpha = abs(C1 * alpha_pos[j] - positions[i][j])
                    X1 = alpha_pos[j] - A1 * D_alpha
                    
                    r1, r2 = random.random(), random.random()
                    A2, C2 = 2 * a * r1 - a, 2 * r2
                    D_beta = abs(C2 * beta_pos[j] - positions[i][j])
                    X2 = beta_pos[j] - A2 * D_beta
                    
                    r1, r2 = random.random(), random.random()
                    A3, C3 = 2 * a * r1 - a, 2 * r2
                    D_delta = abs(C3 * delta_pos[j] - positions[i][j])
                    X3 = delta_pos[j] - A3 * D_delta
                    
                    positions[i][j] = (X1 + X2 + X3) / 3.0
                    
        metrics = {
            "fitness_history": self.fitness_history,
            "alpha_fitness": self.alpha_fitness_history,
            "beta_fitness": self.beta_fitness_history,
            "delta_fitness": self.delta_fitness_history,
            "wolf_distance": self.wolf_distance_history,
            "search_radius": self.search_radius_history
        }
                    
        return alpha_pos, beta_pos, delta_pos, alpha_score, metrics

