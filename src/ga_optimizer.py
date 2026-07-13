import numpy as np
import random
import logging

logger = logging.getLogger(__name__)

class GeneticAlgorithm:
    def __init__(self, config, bounds, memory=None, faci=None):
        self.pop_size = config['population_size']
        self.generations = config['generations']
        self.crossover_rate = config['crossover_rate']
        self.base_mutation_rate = config['mutation_rate']
        self.elite_size = config['elite_size']
        
        self.bounds = bounds
        self.memory = memory
        self.faci = faci
        self.num_genes = len(bounds)
        
        self.population = self._initialize_population()
        self.fitness_history = []
        self.diversity_history = []
        self.mutation_history = []
        self.selection_pressure_history = []
        
    def _initialize_population(self):
        pop = []
        if self.memory:
            elites = self.memory.get_elites_by_faci_similarity(self.faci)
            if elites:
                for p in elites:
                    pop.append(np.array(p))
                
        while len(pop) < self.pop_size:
            ind = [random.uniform(b[0], b[1]) for b in self.bounds]
            pop.append(np.array(ind))
        return np.array(pop[:self.pop_size])
        
    def _calculate_diversity(self):
        if len(self.population) < 2:
            return 0.0
        return np.mean(np.std(self.population, axis=0))
        
    def fitness_function(self, chromosome):
        bt, bert, budget, mask, sem, ent, pri, lr = chromosome
        
        # Estimate expected metrics based on strategy
        macro_f1 = min((bt * 0.4 + bert * 0.4) * (budget / 5.0) + 0.5, 0.95)
        semantic_similarity = sem
        entity_preservation = ent
        diversity_metric = (bt * bert) + (mask * 0.5)
        
        comp_cost = (bt * 0.4 + bert * 0.2) * (budget / 5.0)
        
        cif = self.faci.get('class_imbalance_factor', 1.0) if self.faci else 1.0
        minority_gain = (budget / 5.0) * (1.0 - 1.0/cif) if cif > 1.0 else 0.0
        augmentation_fairness = 1.0 - abs(bt - bert) * 0.5
        
        # Exact Formula:
        # 0.35 MacroF1 + 0.20 Minority Gain + 0.15 Entity Preservation + 
        # 0.10 Semantic Similarity + 0.10 Diversity + 0.10 Augmentation Fairness - 
        # 0.10 Computational Cost
        
        fitness = (
            0.35 * macro_f1 + 
            0.20 * minority_gain + 
            0.15 * entity_preservation + 
            0.10 * semantic_similarity + 
            0.10 * diversity_metric + 
            0.10 * augmentation_fairness - 
            0.10 * comp_cost
        )
        return fitness
        
    def _tournament_selection(self, fitnesses, k=3):
        selected = random.sample(range(self.pop_size), k)
        best = max(selected, key=lambda i: fitnesses[i])
        return self.population[best]
        
    def _crossover(self, p1, p2):
        if random.random() < self.crossover_rate:
            pt = random.randint(1, self.num_genes - 1)
            c1 = np.concatenate((p1[:pt], p2[pt:]))
            c2 = np.concatenate((p2[:pt], p1[pt:]))
            return c1, c2
        return p1.copy(), p2.copy()
        
    def _mutate(self, ind, current_mutation_rate):
        for i in range(self.num_genes):
            if random.random() < current_mutation_rate:
                ind[i] += random.gauss(0, 0.1 * (self.bounds[i][1] - self.bounds[i][0]))
                ind[i] = np.clip(ind[i], self.bounds[i][0], self.bounds[i][1])
        return ind
        
    def optimize(self):
        cif = self.faci.get('class_imbalance_factor', 1.0) if self.faci else 1.0
        
        for gen in range(self.generations):
            fitnesses = [self.fitness_function(ind) for ind in self.population]
            self.fitness_history.append(max(fitnesses))
            
            avg_fit = np.mean(fitnesses)
            max_fit = np.max(fitnesses)
            selection_pressure = max_fit / (avg_fit + 1e-9)
            self.selection_pressure_history.append(selection_pressure)
            
            diversity = self._calculate_diversity()
            self.diversity_history.append(diversity)
            
            target_diversity = 0.2
            
            # Dynamic Mutation Rate based on diversity and class imbalance
            base_mut = self.base_mutation_rate
            if cif > 2.0:
                base_mut *= 1.5 # more exploration for high imbalance
                
            if diversity < target_diversity:
                current_mut_rate = min(base_mut * 2.0, 0.5)
            else:
                current_mut_rate = base_mut
                
            self.mutation_history.append(current_mut_rate)
            
            elite_indices = np.argsort(fitnesses)[-self.elite_size:]
            new_pop = [self.population[i] for i in elite_indices]
            
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_selection(fitnesses)
                p2 = self._tournament_selection(fitnesses)
                c1, c2 = self._crossover(p1, p2)
                new_pop.append(self._mutate(c1, current_mut_rate))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self._mutate(c2, current_mut_rate))
                    
            self.population = np.array(new_pop)
            
        final_fitness = [self.fitness_function(ind) for ind in self.population]
        best_indices = np.argsort(final_fitness)[::-1]
        elite_policies = self.population[best_indices]
        
        metrics = {
            "fitness_history": self.fitness_history,
            "diversity_history": self.diversity_history,
            "mutation_history": self.mutation_history,
            "selection_pressure": self.selection_pressure_history
        }
        
        return elite_policies, metrics

