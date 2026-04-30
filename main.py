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

    """
    execution_stats = []
    for nbiter in range(1,2) : 
        for sbm_ratio in np.arange(0.00, 1.10, 0.10):
            
            G_name = f"artificial_graph_sbmv4_{sbm_ratio:.2f}_pos_{1-sbm_ratio:.2f}_{nbiter}".replace('.', '_')
            G_name_bis = f"artificial_graph_sbmv4_{sbm_ratio:.2f}_pos_{1-sbm_ratio:.2f}_{nbiter}_deepwalk".replace('.', '_')
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
            gp.compute_commus(G, G_name_bis, "deepwalk")            
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
    all_results = gp.analyze_commus(G_name_short = "artificial_graph_sbmv4", nb_iterations=10, spatial_ref = "deepwalk", i_min = 0.00, i_max = 1.00, nb_i=11, name_export_results="deepwalk")
    end_time = time.time()
    duration = end_time - start_time
    print("\n" + "="*50)
    print("📊 TEMPS D'EXEC POUR ANALYSIS PAS HALAL :")
    print("="*50)
    print(f"{duration} secs")

    """
    features_GT_pos = ['pos_dist']
    features_commu_inferee_normal = ["louvain_density"]
    features_commu_inferee_spatial_based_manual_iter = ["spatial_louvain_density"]
    features_commu_inferee_spatial_based_manual_reg = ["spatial_louvain_manualreg_density"]
    features_commu_inferee_spatial_based_scgravity = ["spatial_louvain_scgravity_density"]
    features_commu_inferee_spatial_based_wrdb = ["spatial_louvain_wrdb_density"]
    features_deepwalk = ["deepwalk_dist"]

    experiments = {
        "Inferred_Commu_normal": features_commu_inferee_normal,
        "Inferred_Commu_spatial_manuel_iter": features_commu_inferee_spatial_based_manual_iter,
        "Inferred_Commu_spatial_manuel_reg": features_commu_inferee_spatial_based_manual_reg,
        "Inferred_Commu_spatial_scgravity": features_commu_inferee_spatial_based_scgravity,
        "Inferred_Commu_spatial_wrdb": features_commu_inferee_spatial_based_wrdb,
        "GT_pos": features_GT_pos,
        "GT_pos + Inferred_Commu normal": features_GT_pos + features_commu_inferee_normal,
        "GT_pos + Inferred_Commu spatial manuel iter": features_GT_pos + features_commu_inferee_spatial_based_manual_iter,
        "GT_pos + Inferred_Commu spatial manuel reg": features_GT_pos + features_commu_inferee_spatial_based_manual_reg,
        "GT_pos + Inferred_Commu spatial scgravity": features_GT_pos + features_commu_inferee_spatial_based_scgravity,
        "GT_pos + Inferred_Commu spatial wrdb": features_GT_pos + features_commu_inferee_spatial_based_wrdb,
    }
    """
    "urban_streets_venice",
    "urban_streets_vienna",
    "urban_streets_walnut-creek",
    "urban_streets_ahmedabad",
    "urban_streets_barcelona",
    "urban_streets_bologna",
    "urban_streets_brasilia",
    "urban_streets_cairo",
    "urban_streets_irvine1",
    "urban_streets_irvine2",
    "urban_streets_london",
    "urban_streets_los-angeles",
    "urban_streets_new-delhi",
    "urban_streets_new-york",
    "urban_streets_paris",
    "urban_streets_richmond",
    "urban_streets_san-francisco",

    "urban_streets_savannah",
    "urban_streets_seoul",
    "urban_streets_washington",
    "facebook_organizations_S1",
    "facebook_organizations_S2",
    """
    G_names_list = [
        "facebook_organizations_S2",
        "Airports",
        "fullerene_structures_C1500",
        "urban_streets_savannah",
        "urban_streets_seoul",
        "urban_streets_washington",
        "facebook_organizations_S1",
    ]

    def fix_and_save_graph_metadata(path, path_target, pos_attr='_pos'):
        import io, html, json
        
        # 1. Chargement brut
        print(f"Graphe chargé : {path}")
        with open(path, 'r', encoding='utf-8') as f:
            raw = f.read()
        G = nx.read_graphml(io.StringIO(html.unescape(raw)))
        nodes = list(G.nodes())
        clean_pos_list = []
        
        # 2. Extraction des positions uniquement
        for n_id in nodes:
            val = G.nodes[n_id].get(pos_attr) or G.nodes[n_id].get('pos')
            if val is None:
                raise ValueError(f"❌ Position absente pour {n_id}")
                
            if isinstance(val, str):
                # On nettoie le format texte pour avoir une liste de floats
                s = val.strip().replace('[', '').replace(']', '').replace(',', ' ')
                val = [float(x) for x in s.split()]
            
            clean_pos_list.append([float(x) for x in val])
    
        # 3. MISE À JOUR CIBLÉE (On ne boucle pas sur tous les attributs)
        G.graph['GroundTruth_JSON'] = json.dumps({'GT_pos': clean_pos_list})
    
        # 4. Sauvegarde
        print(f"Graphe sauvegardé : {path}")
        nx.write_graphml(G, path_target, named_key_ids=True)
        
        print(f"✅ MAJ ciblée terminée pour {path}. Les autres attributs n'ont pas été touchés.")

        if 'GroundTruth_JSON' in G.graph:
            print(f"[INIT] Extraction de la GT GroundTruth pour {G_name}...")
            gt_raw = json.loads(G.graph['GroundTruth_JSON'])
            
            GT = {
                'GT_pos': np.array(gt_raw['GT_pos']),
                    }
    
            if 'P_matrix_JSON' in G.graph:
                print("P_matrix trouvée.")
                GT['GT_proba'] = np.array(json.loads(G.graph['P_matrix_JSON']))
        else:
            print("[WARNING] Aucune GroundTruth_JSON trouvée dans G.graph")
            GT = None
    
        return G


    
    for G_name in G_names_list : 
        print("######################################")
        print(f"#### graph {G_name} :  ####")
        print("######################################")
        
        path = f"graph_library/benchmark_graphes_reels/reel_spatial_{G_name}"
        
        try:
            G = load_graphml_safe(f"{path}", startswith = True)
            print(f"Graphe chargé avec succès : {G.number_of_nodes()} nœuds et {G.number_of_edges()} liens.")
        except Exception as e:
            print(f"Erreur lors du chargement de {path} : {e}")

        print(f"Validation 1er nœud : {list(G.nodes(data=True))[0]}")


        for n, data in G.nodes(data=True):
            for attr in ['_pos']:
                if attr in data and isinstance(data[attr], str):
                    try:
                        data[attr] = np.array(ast.literal_eval(data[attr]))
                    except (ValueError, SyntaxError):
                        continue

        first_node = next(iter(G.nodes))
        print(G.nodes[first_node])

        start_time = time.time()
        gp.compute_commus(G, G_name, "GT_pos")            
        end_time = time.time()
        duration = end_time - start_time
    

    gp.analyze_commus(G_name_short="G_reels", nb_iterations=0, spatial_ref = "GT_pos", i_min =0.00, i_max = 1.00, nb_i=11, name_export_results="2026_04_30")
        
    
    


        
        