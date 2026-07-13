import math
from collections import Counter

class FACICalculator:
    FRAUD_KEYWORDS = {"scam", "fraud", "steal", "stolen", "unauthorized", "phishing", "fake"}
    RISK_INDICATORS = {"stolen card", "large amount", "hacked", "breach"}
    AMBIGUITY_INDICATORS = {"maybe", "not sure", "possibly", "might"}
    
    def compute(self, text, entity_density, class_imbalance_factor=1.0):
        """
        Computes the Fraud-Aware Complexity Index (FACI) in a Multi-Dimensional format.
        """
        tokens = text.lower().split()
        N = max(len(tokens), 1)
        
        complexity = min(len(tokens) / 500.0, 1.0)
        e_density = min(entity_density, 1.0)
        
        fraud_count = sum(1 for w in tokens if w in self.FRAUD_KEYWORDS)
        fraud_density = min(fraud_count / N, 1.0) * 5.0
        fraud_density = min(fraud_density, 1.0)
        
        risk_count = sum(1 for r in self.RISK_INDICATORS if r in text.lower())
        risk_score = min(risk_count / 2.0, 1.0)
        
        amb_count = sum(1 for a in self.AMBIGUITY_INDICATORS if a in text.lower())
        ambiguity = min(amb_count / 3.0, 1.0)
        
        rare_word_density = min(len([t for t in tokens if len(t) > 8]) / N * 3.0, 1.0)
        
        redaction_count = sum(1 for t in tokens if 'xxxx' in t)
        redaction_density = min(redaction_count / N, 1.0)
        
        # Rigorous Semantic Entropy (Shannon Entropy normalized)
        token_counts = Counter(tokens)
        entropy = -sum((count / N) * math.log2(count / N) for count in token_counts.values())
        max_entropy = math.log2(N) if N > 1 else 1.0
        semantic_entropy = min(entropy / max_entropy, 1.0)
        
        w_C, w_E, w_F, w_R, w_A = 0.2, 0.25, 0.3, 0.15, 0.1
        faci_scalar = (w_C * complexity) + (w_E * e_density) + (w_F * fraud_density) + (w_R * risk_score) + (w_A * ambiguity)
        
        if faci_scalar < 0.30:
            faci_budget = 1
        elif faci_scalar <= 0.60:
            faci_budget = 2
        else:
            faci_budget = 4
            
        recommended_budget = int(faci_budget * class_imbalance_factor)
        recommended_budget = max(1, recommended_budget)
            
        return {
            "scalar": faci_scalar,
            "complexity": complexity,
            "entity_density": e_density,
            "fraud_density": fraud_density,
            "risk_score": risk_score,
            "ambiguity": ambiguity,
            "semantic_entropy": semantic_entropy,
            "rare_word_density": rare_word_density,
            "redaction_density": redaction_density,
            "class_imbalance_factor": class_imbalance_factor,
            "recommended_budget": recommended_budget
        }

