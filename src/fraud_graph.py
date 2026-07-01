import networkx as nx

class FraudGraphBuilder:
    def build(self, entities):
        G = nx.DiGraph()
        G.add_node("Victim", role="person")
        
        for idx, ent in enumerate(entities):
            node_id = f"{ent['type']}_{idx}"
            G.add_node(node_id, **ent)
            
            if ent['type'] == 'OTP':
                G.add_edge("Victim", node_id, rel="provided")
                G.add_node("Action_shared", type="action")
                G.add_edge(node_id, "Action_shared", rel="enables")
            elif ent['type'] == 'CARD':
                G.add_edge("Victim", node_id, rel="owns")
            elif ent['type'] == 'MONEY':
                G.add_edge("Victim", node_id, rel="lost")
            else:
                G.add_edge("Victim", node_id, rel="related_to")
                
        return G

    def get_entity_density(self, G, text_length):
        num_entities = G.number_of_nodes() - 1
        if text_length == 0:
            return 0.0
        return num_entities / text_length
