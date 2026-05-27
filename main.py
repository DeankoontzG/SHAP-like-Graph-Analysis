import src.SHAP_like_graph_tool as gp
import src.SHAP_like_graph_tool.utils as gput

import networkx as nx
import html
import io
import ast
import json
import glob
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
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBClassifier
import optuna
import time

# nohup python -u main.py 2>&1 | grep --line-buffered -vE "it/s|%|\[.*\]|^----" | grep --line-buffered "." > myoutfile &
# 

def load_graphml_safe(path, startswith = False):
        if startswith : 
            search_pattern = f"{path}*.graphml"
            files = glob.glob(search_pattern)
        
            if not files:
                print(f"❌ Aucun fichier trouvé pour : {search_pattern}")
                return None
        
            # On prend le premier fichier trouvé
            path = files[0]
            print(f"📂 Fichier trouvé : {path}")
    
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

    gp.computeShapValsGTforAllGraphs()

    """
    execution_stats = []
    for nbiter in range(2,11) : 
        for sbm_ratio in np.arange(0.00, 1.10, 0.10):
            
            G_name = f"artificial_graph_sbmv_4_{sbm_ratio:.2f}_pos_{1-sbm_ratio:.2f}_{nbiter}".replace('.', '_')
            G_name_bis = f"artificial_graph_sbmv_4_AllBasicMetrics_{sbm_ratio:.2f}_pos_{1-sbm_ratio:.2f}_{nbiter}".replace('.', '_')
            print("######################################")
            print(f"#### graph {G_name} :  ####")
            print("######################################")
            
            path = f"graph_library/{G_name}.graphml"
            
            try:
                G = load_graphml_safe(path)
                print(f"Graphe chargé avec succès : {G.number_of_nodes()} nœuds et {G.number_of_edges()} liens.")
            except Exception as e:
                print(f"Erreur lors du chargement de {path} : {e}")
            
            start_time = time.time()
            gp.compute_commus(G, G_name_bis, "GT_pos", computeEmb=True)      
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
    
    start_time = time.time()
    all_results = gp.analyze_commus(G_name_short = "artificial_graph_sbmv_6_comuAndEmb", nb_iterations=10, spatial_ref = "GT_pos", i_min = 0.00, i_max = 1.00, nb_i=11, name_export_results="GT_pos")
    end_time = time.time()
    duration = end_time - start_time
    print("\n" + "="*50)
    print("📊 TEMPS D'EXEC POUR ANALYSIS PAS HALAL :")
    print("="*50)
    print(f"{duration} secs")
    """

    
   
    #"fullerene_structures_C1500",
    #"urban_streets_savannah",
    #"urban_streets_seoul",
    #"urban_streets_washington",
    #"facebook_organizations_S1",
    #"facebook_organizations_S2",
    
    
    
    """
    G_names_list = [
        "Airports",
        #"eu_airlines",
        #"faa_routes"
    ]
            
    
    for iteration in range (0,1):
        for G_name in G_names_list : 
            G_name_bis = f"{G_name}_test_{iteration}"
            print("######################################")
            print(f"#### graph {G_name_bis} :  ####")
            print("######################################")

            path = f"graph_library/benchmark_graphes_reels/reel_spatial_{G_name}"
            
            try:
                G = load_graphml_safe(f"{path}", startswith = True)
                print(f"Graphe chargé avec succès : {G.number_of_nodes()} nœuds et {G.number_of_edges()} liens.")
            except Exception as e:
                print(f"Erreur lors du chargement de {path} : {e}")
    
            print(f"Validation 1er nœud : {list(G.nodes(data=True))[0]}")
    
            
            for n, data in G.nodes(data=True):
                for attr in ['GT_pos']:
                    if attr in data and isinstance(data[attr], str):
                        try:
                            data[attr] = np.array(ast.literal_eval(data[attr]))
                        except (ValueError, SyntaxError):
                            continue
            
            if 'GroundTruth_JSON' in G.graph:
                print(f"[INIT] Extraction de la GT GroundTruth pour {G_name}...")
                gt_raw = json.loads(G.graph['GroundTruth_JSON'])
                
                GT = {
                    'GT_pos': np.array(gt_raw['GT_pos']),
                     }
            else:
                print("[WARNING] Aucune GroundTruth_JSON trouvée dans G.graph")
                GT = None
    
            if GT is not None and 'GT_pos' in GT:
                for i, node_id in enumerate(G.nodes()):
                    G.nodes[node_id]['GT_pos'] = GT['GT_pos'][i]
    
            first_node = next(iter(G.nodes))
            print(G.nodes[first_node])

            start_time = time.time()
            gp.compute_commus_greels(G, G_name_bis, "GT_pos")      
            end_time = time.time()
            duration = end_time - start_time
                
            print(f"⏱️ Terminé en {duration:.2f} secondes.")


            dataset_train = gp.load_dataset(filename=f"dataset_train_{G_name_bis}")
            dataset_hidden = gp.load_dataset(filename=f"dataset_hidden_{G_name_bis}")

            exclude = ['u', 'v', 'target', 'label', 'cn', 'aa', 'ra', 'jc', 'pa', 'sp',
                      #'louvain_density', 'spatial_louvain_density', 'spatial_louvain_manualiter_0_20_density', 'spatial_louvain_manualiter_0_50_density', 'spatial_louvain_manualiter_0_80_density'
                      ]
            features = [col for col in dataset_train.columns if (col not in exclude)]
            print(f"FEATURES pour XGBOOST : {features}")
            best_params = {
                'n_estimators': 1000, # On met beaucoup
                'learning_rate': 0.05,
                'max_depth': 5,
                'min_child_weight': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'tree_method': 'hist',
                'n_jobs': -1,
                'random_state': 42
            }
            
            results_test, model, X_train, y_train, X_test, y_test = gp.train_and_test_xgboost(dataset_train, features=features, parameters=best_params)
            
            X_hidden = dataset_hidden[features] if features else dataset_hidden.drop(["target", "u", "v", "label"], axis=1)
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
            gp.loadsave_data_joblib(data=data_to_save, filename=f"xgboost_data_{G_name_bis}.joblib", mode="save")
        
            print("\n RÉSULTATS")
            print(results_test_hidden.to_string(index=False))
    
            print("\n [SHAP] Shapley va ! Lu.")
    
            shap_explanation = gp.analyze_with_shap_tree(model, X_hidden, y_hidden)
            print("Sauvegarde de l'analyse SHAP")
            gp.loadsave_data_joblib(data=shap_explanation, filename=f"shap_explainer_{G_name_bis}.joblib", mode="save")

    
            start_time = time.time()
            gp.compute_commus_greels(G, G_name_bis, "GT_pos")            
            end_time = time.time()
            duration = end_time - start_time
            """
        

            #gp.analyze_commus_greels(G_name_short="G_reels", nb_iterations=0, spatial_ref = "GT_pos", i_min =0.00, i_max = 1.00, nb_i=11, name_export_results="2026_05_11")