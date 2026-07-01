import math

class FACICalculator:
    FRAUD_KEYWORDS = {"scam", "fraud", "steal", "stolen", "unauthorized", "phishing", "fake"}
    RISK_INDICATORS = {"stolen card", "large amount", "hacked", "breach"}
    AMBIGUITY_INDICATORS = {"maybe", "not sure", "possibly", "might"}
    
    def compute(self, text, entity_density):
        tokens = text.lower().split()
        
        complexity = min(len(tokens) / 500.0, 1.0)
        e_density = min(entity_density, 1.0)
        
        fraud_count = sum(1 for w in tokens if w in self.FRAUD_KEYWORDS)
        fraud_density = min(fraud_count / max(len(tokens), 1), 1.0) * 5.0
        fraud_density = min(fraud_density, 1.0)
        
        risk_count = sum(1 for r in self.RISK_INDICATORS if r in text.lower())
        risk_score = min(risk_count / 2.0, 1.0)
        
        amb_count = sum(1 for a in self.AMBIGUITY_INDICATORS if a in text.lower())
        ambiguity = min(amb_count / 3.0, 1.0)
        
        # Heuristics for new dimensions
        rare_word_density = min(len([t for t in tokens if len(t) > 8]) / max(len(tokens), 1) * 3.0, 1.0)
        
        # Simple entropy approximation based on unique tokens
        unique_tokens = len(set(tokens))
        semantic_entropy = min(unique_tokens / max(len(tokens), 1), 1.0)
        
        # Base scalar calculation for budgeting
        w_C, w_E, w_F, w_R, w_A = 0.2, 0.25, 0.3, 0.15, 0.1
        faci_scalar = (w_C * complexity) + (w_E * e_density) + (w_F * fraud_density) + (w_R * risk_score) + (w_A * ambiguity)
        
        if faci_scalar < 0.30:
            recommended_budget = 1
        elif faci_scalar <= 0.60:
            recommended_budget = 2
        else:
            recommended_budget = 4
            
        return {
            "scalar": faci_scalar,
            "complexity": complexity,
            "entity_density": e_density,
            "fraud_density": fraud_density,
            "risk_score": risk_score,
            "ambiguity": ambiguity,
            "rare_word_density": rare_word_density,
            "semantic_entropy": semantic_entropy,
            "recommended_budget": recommended_budget
        }
