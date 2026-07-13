import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.faci import FACICalculator

def test_faci_dynamic_variation():
    calc = FACICalculator()
    
    # 5 highly varying strings
    texts = [
        "A scammer stole my card and took a large amount of money. I think my account was hacked and there was a phishing breach.",
        "Simple complaint. I didn't get my statement this month. Please send it.",
        "maybe I am not sure what happened, possibly someone used my card, might be stolen.",
        "Scam fraud steal unauthorized phishing fake stolen card large amount hacked breach maybe not sure.",
        "A perfectly normal transaction that I am inquiring about regarding the recent change in my interest rates which seem excessively high compared to the federal reserve benchmark."
    ]
    
    scalars = []
    budgets = []
    
    for text in texts:
        res = calc.compute(text, entity_density=0.1)
        scalars.append(res['scalar'])
        budgets.append(res['recommended_budget'])
        
    # Assert they are not all identical
    assert len(set(scalars)) > 1, "FACI scalars should vary based on text complexity!"
    print(f"Scalars: {scalars}")
    print(f"Budgets: {budgets}")
