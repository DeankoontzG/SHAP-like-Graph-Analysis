import json
import html
import io
import numpy as np
import networkx as nx
import pandas as pd
from src.SHAP_like_graph_tool import user_usable_functions as gp
from src.SHAP_like_graph_tool import utils 


if __name__ == "__main__":

    def load_graphml_safe(path):
        with open(path, 'r', encoding='utf-8') as f:
            raw_data = f.read()

        clean_data = html.unescape(raw_data)
        G = nx.read_graphml(io.StringIO(clean_data))
        
        print(f"✅ Graphe chargé : {G.number_of_nodes()} nœuds et {G.number_of_edges()} liens.")
        return G

    G = load_graphml_safe("graph_library/reel_Airports.graphml")
    
    gp.execute(G, "reel_Airports")