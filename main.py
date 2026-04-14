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

    features = ['ra', 'pr_u', 'pr_v', 'lcc_u', 'lcc_v', 'katz_u', 'katz_v', 'surprise_density', 'same_surprise', 'deepwalk_cos', 'deepwalk_dist', 'deepwalk_rank']

    for sbm_ratio in np.arange(1.00, -0.25, -0.25):

        G_name = f"artificial_graph_sbmv5_{sbm_ratio:.2f}_pos_{1-sbm_ratio:.2f}".replace('.', '_')   
        G_name_ter = f"artificial_graph_sbmv5_{sbm_ratio:.2f}_pos_{1-sbm_ratio:.2f}_reduced".replace('.', '_')  
        print("######################################")
        print(f"#### graph {G_name_ter} :  ####")
        print("######################################")

        path = f"graph_library/{G_name}.graphml"
        
        try:
            G = load_graphml_safe(path)
            print(f"Graphe chargé avec succès : {G.number_of_nodes()} nœuds et {G.number_of_edges()} liens.")
        except Exception as e:
            print(f"Erreur lors du chargement de {path} : {e}")

        G_kept, G_hidden = gput.hide_graph_links(G, test_size=0.10)
        G_train, G_test = gput.hide_graph_links(G_kept, test_size=0.15)
    
        best_params, results_summary = gput.k_fold_cross_validation(G_kept, k=1, features_list=features, n_trials=50, GroundTruth=None, graph_name= G_name_ter)
        print(best_params)

        dataset_train = gput.load_dataset(f"dataset_train_{G_name}")
        dataset_hidden = gput.load_dataset(f"dataset_hidden_{G_name}")

        results_test, model, X_train, y_train, X_test, y_test = gp.train_and_test_xgboost(dataset_train, features=features, parameters=best_params)
        
        X_hidden = dataset_hidden[features]
        y_hidden = dataset_hidden['target']
        
        results_hidden = gp.get_performance_metrics(model, X_hidden, y_hidden, "Hidden_")
        results_test_hidden = pd.concat([results_test, results_hidden], axis=1)
    
        data_to_save = {
            "results": results_test_hidden,
            "model": model,
            "X_test": X_test,
            "X_train": X_train,
            "y_test": y_test,
            "y_train": y_train,
            "X_hidden": X_hidden,
            "y_hidden": y_hidden,
            "best_params": best_params
        }

        print("[PREP] Sauvegarde des données XGBoost (model, X/y Test et Hidden)")
        gput.loadsave_data_joblib(data=data_to_save, filename=f"xgboost_data_{G_name_ter}.joblib", mode="save")
    
        print("\n RÉSULTATS")
        print(results_test_hidden.to_string(index=False))

        shap_explanation = gput.analyze_with_shap_tree(model, X_hidden, y_hidden)
        print("Sauvegarde de l'analyse SHAP")
        gput.loadsave_data_joblib(data=shap_explanation, filename=f"shap_explainer_{G_name_ter}.joblib", mode="save")


"""
if __name__ == "__main__":

    execution_stats = []

    for sbm_ratio in np.arange(1.00, -0.25, -0.25):
        
        G_name = f"artificial_graph_sbmv5_{sbm_ratio:.2f}_pos_{1-sbm_ratio:.2f}".replace('.', '_')   
        G_name_bis = f"artificial_graph_sbmv5_{sbm_ratio:.2f}_pos_{1-sbm_ratio:.2f}_GT".replace('.', '_')  
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

    for sbm_ratio in np.arange(1.00, -0.25, -0.25):
        
        G_name = f"artificial_graph_sbmv4_{sbm_ratio:.2f}_pos_{1-sbm_ratio:.2f}".replace('.', '_')   
        G_name_bis = f"artificial_graph_sbmv4_{sbm_ratio:.2f}_pos_{1-sbm_ratio:.2f}_GT".replace('.', '_')  
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