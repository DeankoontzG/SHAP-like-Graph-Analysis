import json
import html
import io
import numpy as np
import networkx as nx
import pandas as pd
from src.SHAP_like_graph_tool import user_usable_functions as gp
from src.SHAP_like_graph_tool import utils 


if __name__ == "__main__":

    for i in np.arange(0.00, 1.25, 0.25):
        print("######################################")
        print(f"#### graph sbmv2 {i:.2f} pos {1-i:.2f} ####")
        print("######################################")
        G_name = f"artificial_graph_sbmv2_{f'{i:.2f}'.replace('.', '_')}_pos_{f'{1-i:.2f}'.replace('.', '_')}"
        print(G_name)
    
        with open(f"graph_library/{G_name}.json", 'r', encoding='utf-8') as f:
            data = json.load(f)  
    
        try:
            G = nx.node_link_graph(data)
            print(f"Graphe chargé avec succès : {G.number_of_nodes()} nœuds et {G.number_of_edges()} liens.")
        except Exception as e:
            print(f"Erreur lors de la conversion : {e}")
    
        gp.execute(G, G_name, steps=["prep"])