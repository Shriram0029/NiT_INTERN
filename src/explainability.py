import logging

class ExplainabilityEngine:
    def __init__(self):
        self.logger = logging.getLogger("ExplainabilityEngine")
        
    def _categorize(self, value):
        if value > 0.66:
            return "High"
        elif value > 0.33:
            return "Medium"
        return "Low"
        
    def generate_explanation(self, chunk_id, faci_dict, prediction):
        if not faci_dict or not prediction:
            return f"Chunk {chunk_id:02d} | No context available for explanation."
            
        strategy = prediction.get('strategy', 'Unknown')
        f1_increase = prediction.get('expected_macro_f1', 0.0) * 100
        
        complexity = self._categorize(faci_dict.get("complexity", 0.0))
        fraud_density = self._categorize(faci_dict.get("fraud_density", 0.0))
        ambiguity = self._categorize(faci_dict.get("ambiguity", 0.0))
        
        explanation = (
            f"Chunk {chunk_id:02d}\n"
            f"FACI: {fraud_density} Fraud Density, {complexity} Complexity, {ambiguity} Ambiguity\n"
            f"↓\n"
            f"Selected: {strategy}\n"
            f"↓\n"
            f"Reason: Historical utility on similar complaints is expected to yield Macro F1 around {f1_increase:.1f}%."
        )
        
        return explanation

