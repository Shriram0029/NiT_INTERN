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
        
    def _determine_strategy(self, p_bt, p_bert):
        if p_bt == 0 and p_bert == 0:
            return "No Augmentation"
        elif p_bt > 0.5 and p_bert > 0.5:
            return "Hybrid"
        elif p_bt > p_bert:
            return "Back Translation"
        else:
            return "BERT Contextual"

    def optimize(self, chunk_id, faci_features):
        logger.info(f"Starting Hybrid GA-GWO Optimization for Chunk {chunk_id}")
        
        # Phase 1: Global Exploration via GA
        ga = GeneticAlgorithm(self.ga_config, self.bounds, self.memory, faci_features)
        elite_policies, ga_metrics = ga.optimize()
        
        logger.info(f"GA Optimization complete. Best GA Fitness: {ga_metrics['fitness_history'][-1]:.4f}")
        
        # Phase 2: Local Exploitation via GWO
        gwo = GWOOptimizer(self.gwo_config, self.bounds, self.memory)
        alpha, beta, delta, alpha_score, gwo_metrics = gwo.optimize(elite_policies, ga.fitness_function)
        
        logger.info(f"GWO Optimization complete. Best Alpha Fitness: {alpha_score:.4f}")
        
        # Phase 3: True Augmentation Selection (Prediction)
        bt, bert, budget_raw, mask, sem, ent, pri, lr = alpha
        
        budget = int(round(budget_raw))
        strategy = self._determine_strategy(bt, bert)
        
        # If strategy is "No Augmentation", force budget to 0
        if strategy == "No Augmentation":
            budget = 0
            
        expected_macro_f1 = min((bt * 0.4 + bert * 0.4) * (budget_raw / 5.0) + 0.5, 0.95)
        expected_cost = (bt * 0.4 + bert * 0.2) * (budget_raw / 5.0)
        expected_utility = alpha_score
        confidence = min(alpha_score / 1.5, 1.0) if alpha_score > 0 else 0.0
        
        reason = f"Based on FACI profile, GA-GWO selected {strategy} (Budget: {budget}) aiming for Expected Utility {expected_utility:.4f} and F1 {expected_macro_f1:.4f}."
        
        prediction = {
            "strategy": strategy,
            "budget": budget,
            "expected_utility": expected_utility,
            "expected_cost": expected_cost,
            "expected_macro_f1": expected_macro_f1,
            "confidence": confidence,
            "reason": reason
        }
        
        return {
            "policy": alpha,
            "prediction": prediction,
            "alpha": alpha,
            "beta": beta,
            "delta": delta,
            "elite_policies": elite_policies.tolist(),
            "ga_metrics": ga_metrics,
            "gwo_metrics": gwo_metrics,
            "fitness": alpha_score
        }

