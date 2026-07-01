from src.ga_optimizer import GeneticAlgorithm
from src.gwo_optimizer import GWOOptimizer
import logging

logger = logging.getLogger(__name__)

class HybridOptimizer:
    def __init__(self, ga_config, gwo_config, bounds, memory=None):
        self.ga_config = ga_config
        self.gwo_config = gwo_config
        self.bounds = bounds
        self.memory = memory
        
    def optimize(self, chunk_id, faci_features):
        logger.info(f"Starting Hybrid GA-GWO Optimization for Chunk {chunk_id}")
        
        ga = GeneticAlgorithm(self.ga_config, self.bounds, self.memory, faci_features)
        elite_policies, ga_fit_hist, ga_div_hist, ga_mut_hist = ga.optimize()
        
        logger.info(f"GA Optimization complete. Best GA Fitness: {ga_fit_hist[-1]:.4f}")
        
        gwo = GWOOptimizer(self.gwo_config, self.bounds, self.memory)
        alpha, beta, delta, alpha_score = gwo.optimize(elite_policies, ga.fitness_function)
        
        logger.info(f"GWO Optimization complete. Best Alpha Fitness: {alpha_score:.4f}")
        
        return {
            "policy": alpha,
            "alpha": alpha,
            "beta": beta,
            "delta": delta,
            "elite_policies": elite_policies.tolist(),
            "ga_history": ga_fit_hist,
            "ga_diversity": ga_div_hist,
            "ga_mutation": ga_mut_hist,
            "gwo_history": gwo.fitness_history,
            "fitness": alpha_score
        }
