import src.SHAP_like_graph_tool as gp
import src.SHAP_like_graph_tool.utils as gput

import networkx as nx
import html
import io
import ast
import json
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
from re import X
import numpy as np
import pandas as pd
import networkx as nx
import itertools
from math import factorial
from networkx.algorithms.community import louvain_communities
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold, train_test_split
from node2vec import Node2Vec
from infomap import Infomap
import graph_tool.all as gt
import shap
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBClassifier
import optuna
import time

# nohup python -u main.py 2>&1 | grep --line-buffered -vE "it/s|%|\[.*\]" | grep --line-buffered "." > myoutfile &
# 

def load_graphml_safe(path):
        with open(path, 'r', encoding='utf-8') as f:
            raw_data = f.read()

        clean_data = html.unescape(raw_data)
        G = nx.read_graphml(io.StringIO(clean_data))

        for n, data in G.nodes(data=True):
            if 'GT_pos' in data and isinstance(data['GT_pos'], str):
                data['GT_pos'] = np.array(ast.literal_eval(data['GT_pos']))
        
        print(f"✅ Graphe chargé : {G.number_of_nodes()} nœuds et {G.number_of_edges()} liens.")
        return G
    
def save_as_graphml(G_nx, filename="mon_graphe.graphml", folder="graph_library"):
    path = os.path.join(folder, filename)
    nx.write_graphml(G_nx, path)
    print(f"Graphe exporté avec succès dans : {path}")


if __name__ == "__main__":

    execution_stats = []
    for sbm_ratio in np.arange(1.00, -0.25, -0.25):
        
        G_name = f"artificial_graph_sbmv4_{sbm_ratio:.2f}_pos_{1-sbm_ratio:.2f}".replace('.', '_')   
        G_name_bis = f"artificial_graph_sbmv4_{sbm_ratio:.2f}_pos_{1-sbm_ratio:.2f}".replace('.', '_')  
        print("######################################")
        print(f"#### graph {G_name_bis} :  ####")
        print("######################################")
        
        path = f"graph_library/{G_name}.graphml"
        
        try:
            G = load_graphml_safe(path)
            print(f"Graphe chargé avec succès : {G.number_of_nodes()} nœuds et {G.number_of_edges()} liens.")
        except Exception as e:
            print(f"Erreur lors du chargement de {path} : {e}")
        
        start_time = time.time()
        gp.execute(G, G_name_bis, add_P_matrix = True)                
        end_time = time.time()
        duration = end_time - start_time

        execution_stats.append({
                "Graph": G_name,
                "Nodes": G.number_of_nodes(),
                "Edges": G.number_of_edges(),
                "Time_sec": round(duration, 2),
                "Time_per_node": round(duration / G.number_of_nodes(), 4) if G.number_of_nodes() > 0 else 0,
                "Time_per_link": round(duration / G.number_of_edges(), 4) if G.number_of_edges() > 0 else 0
            })
            
        print(f"⏱️ Terminé en {duration:.2f} secondes.")


    df = pd.DataFrame(execution_stats)
    print("\n" + "="*50)
    print("📊 RÉSUMÉ DES STATISTIQUES D'EXÉCUTION")
    print("="*50)
    print(df.to_string(index=False))
    

    """
    for sbm_ratio in np.arange(1.00, -0.25, -0.25):
        
        G_name = f"artificial_graph_sbmv5_{sbm_ratio:.2f}_pos_{1-sbm_ratio:.2f}".replace('.', '_')   
        G_name_bis = f"artificial_graph_sbmv5_{sbm_ratio:.2f}_pos_{1-sbm_ratio:.2f}".replace('.', '_')  
        print("######################################")
        print(f"#### graph {G_name_bis} :  ####")
        print("######################################")
        
        path = f"graph_library/{G_name}.graphml"
        
        try:
            G = load_graphml_safe(path)
            print(f"Graphe chargé avec succès : {G.number_of_nodes()} nœuds et {G.number_of_edges()} liens.")
        except Exception as e:
            print(f"Erreur lors du chargement de {path} : {e}")
        
        start_time = time.time()
        gp.execute(G, G_name_bis, add_P_matrix = True)                
        end_time = time.time()
        duration = end_time - start_time

        execution_stats.append({
                "Graph": G_name,
                "Nodes": G.number_of_nodes(),
                "Edges": G.number_of_edges(),
                "Time_sec": round(duration, 2),
                "Time_per_node": round(duration / G.number_of_nodes(), 4) if G.number_of_nodes() > 0 else 0,
                "Time_per_link": round(duration / G.number_of_edges(), 4) if G.number_of_edges() > 0 else 0
            })
            
        print(f"⏱️ Terminé en {duration:.2f} secondes.")


    df = pd.DataFrame(execution_stats)
    print("\n" + "="*50)
    print("📊 RÉSUMÉ DES STATISTIQUES D'EXÉCUTION")
    print("="*50)
    print(df.to_string(index=False))

    execution_stats = []
    """

    