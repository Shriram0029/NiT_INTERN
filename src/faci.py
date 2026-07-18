import math
from collections import Counter

class FACICalculator:
    FRAUD_KEYWORDS = {"scam", "fraud", "steal", "stolen", "unauthorized", "phishing", "fake", "recognize", "identity"}
    RISK_INDICATORS = {"stolen card", "large amount", "hacked", "breach"}
    AMBIGUITY_INDICATORS = {"maybe", "not sure", "possibly", "might"}
    
    def compute(self, text, entity_density=0.0):
        """
        Computes the Fraud-Aware Complexity Index (FACI) in a Multi-Dimensional format.
        Returns a dictionary of normalized features and a numerical vector for memory.
        """
        tokens = text.lower().split()
        N = max(len(tokens), 1)
        
        # 1. Complexity (Normalized to 0-1)
        complexity = min(len(tokens) / 300.0, 1.0)
        
        # 2. Entity Density (Assuming it's passed as a ratio 0-1)
        e_density = min(entity_density, 1.0)
        
        # 3. Fraud Density (Normalized)
        fraud_count = sum(1 for w in tokens if w in self.FRAUD_KEYWORDS)
        fraud_density = min((fraud_count / N) * 5.0, 1.0)
        
        # 4. Risk Score (Normalized)
        risk_count = sum(1 for r in self.RISK_INDICATORS if r in text.lower())
        risk_score = min(risk_count / 2.0, 1.0)
        
        # 5. Ambiguity (Normalized)
        amb_count = sum(1 for a in self.AMBIGUITY_INDICATORS if a in text.lower())
        ambiguity = min(amb_count / 3.0, 1.0)
        
        # 6. Rare Word Density (Normalized)
        rare_words = sum(1 for t in tokens if len(t) > 8)
        rare_word_density = min((rare_words / N) * 3.0, 1.0)
        
        # 7. Redaction Density (Normalized)
        redaction_count = sum(1 for t in tokens if 'xxxx' in t)
        redaction_density = min(redaction_count / N, 1.0)
        
        # 8. Semantic Entropy (Normalized Shannon Entropy)
        token_counts = Counter(tokens)
        entropy = -sum((count / N) * math.log2(count / N) for count in token_counts.values())
        max_entropy = math.log2(N) if N > 1 else 1.0
        semantic_entropy = min(entropy / max_entropy, 1.0)
        
        faci_vector = [
            complexity, e_density, fraud_density, risk_score, 
            ambiguity, semantic_entropy, rare_word_density, redaction_density
        ]
        
        # Calculate scalar for rough budgeting fallback
        w_C, w_E, w_F, w_R, w_A = 0.2, 0.2, 0.2, 0.2, 0.2
        faci_scalar = (w_C * complexity) + (w_E * e_density) + (w_F * fraud_density) + (w_R * risk_score) + (w_A * ambiguity)
        
        return {
            "scalar": faci_scalar,
            "vector": faci_vector,
            "complexity": complexity,
            "entity_density": e_density,
            "fraud_density": fraud_density,
            "risk_score": risk_score,
            "ambiguity": ambiguity,
            "semantic_entropy": semantic_entropy,
            "rare_word_density": rare_word_density,
            "redaction_density": redaction_density
        }
