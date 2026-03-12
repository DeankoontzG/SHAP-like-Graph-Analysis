import src.SHAP_like_graph_tool as gp
import src.SHAP_like_graph_tool.utils as gput

import networkx as nx
import html
import io
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

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
        G = nx.node_link_graph(data)
    
    print(f"Graphe chargé : {G.number_of_nodes()} nœuds et {G.number_of_edges()} liens.")
    return G

def load_graphml_safe(path):
        with open(path, 'r', encoding='utf-8') as f:
            raw_data = f.read()

        clean_data = html.unescape(raw_data)
        G = nx.read_graphml(io.StringIO(clean_data))
        
        print(f"✅ Graphe chargé : {G.number_of_nodes()} nœuds et {G.number_of_edges()} liens.")
        return G

def add_ground_truth_groups(df, G):
    true_partition = dict(nx.get_node_attributes(G, 'block'))
    
    if not true_partition:
        print("Dictionnaire vide ! Vérifie si 'block' existe dans G.nodes(data=True)")
        return df

    df['group_u'] = df['u'].map(true_partition)
    df['group_v'] = df['v'].map(true_partition)
    
    df['group_u'] = df['group_u'].fillna(-1).astype(int)
    df['group_v'] = df['group_v'].fillna(-1).astype(int)
    df['same_group'] = (df['group_u'] == df['group_v']).astype(int)
    
    return df

def shuffle_ground_truth(df):
    """
    Crée des versions mélangées des colonnes de groupes.
    """
    df_shuffled = df.copy()
    # On récupère tous les IDs uniques présents et on les mélange
    unique_groups = df_shuffled['group_u'].unique()
    shuffled_map = dict(zip(unique_groups, np.random.permutation(unique_groups)))
    
    # On applique le mapping pour que la structure logique "same_group" disparaisse
    df_shuffled['group_u_shuffled'] = df_shuffled['group_u'].map(shuffled_map)
    df_shuffled['group_v_shuffled'] = df_shuffled['group_v'].map(shuffled_map)
    # On recalcule un same_group qui ne veut plus rien dire
    df_shuffled['same_group_shuffled'] = (df_shuffled['group_u_shuffled'] == df_shuffled['group_v_shuffled']).astype(int)
    
    return df_shuffled

def k_fold_cross_validation_exp(G_train, folds_data, features_list=None, n_trials=50, graph_name="G_NAME"):
    study = gput._run_optuna_tuning(folds_data, features_list, n_trials=n_trials)
    
    results = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            results.append({
                'Trial': trial.number,
                'Avg_AUC': trial.value,
                'Std_AUC': trial.user_attrs.get('std_auc'),
                'Avg_AP': trial.user_attrs.get('avg_ap'),
                'Delta_AUC': trial.user_attrs.get('delta_auc'),
                'Params': trial.params
            })
    
    summary_df = pd.DataFrame(results).sort_values(by='Avg_AUC', ascending=False)

    print("\n" + "="*80)
    print(f"{'RÉSULTATS OPTUNA : BASELINE VS TOP CONFIGURATIONS':^80}")
    print("="*80)

    cols = ['Trial', 'Avg_AUC', 'Std_AUC', 'Avg_AP', 'Delta_AUC']
    print(summary_df[summary_df['Trial'] == 0][cols].to_string(index=False))
    print("-" * 80)
    print(summary_df.head(10)[cols].to_string(index=False))
    print("="*80)

    save_dir = "outputs/results"
    os.makedirs(save_dir, exist_ok=True)
    filename = f"optuna_results_{graph_name}.csv"
    full_path = os.path.join(save_dir, filename)
    summary_df.to_csv(full_path, index=False)
    print(f"Résultats sauvegardés dans : {full_path}")

    best_params = study.best_params.copy()
    best_params.update({'tree_method': 'hist', 'n_estimators': 150})
    
    return best_params, summary_df


if __name__ == "__main__":

    
    features_ground_truth = ['block_reel_u', 'block_reel_v','same_block_reel', "dist_reelle", "proba_lien_reelle"]
    features_structure = ['pa', 'pr_u', 'pr_v', 'lcc_u', 'lcc_v', 'and_u', 'and_v', 'dc_u',
                          'sp', 
                          'jc', 'aa', 'cn',
                          'dc_v']
    features_commu_inferee = ['louvain_density', 'same_louvain', 'infomap_density', 'same_infomap', 'sbm_density', 'same_sbm']
    features_commu_inferee_same_group_only = ['same_louvain', 'same_infomap', 'same_sbm']
    features_embeddings = ["deepwalk_dist", "deepwalk_cos", "n2v_homophily_dist", "n2v_homophily_cos"]
    
    experiments = {
        #"Ground_Truth_Only": features_ground_truth,
        "Structure_Only": features_structure,
        "Embeddings_Only": features_embeddings,
        "Inferred_Commu only": features_commu_inferee,
        "Embeddings + Inferred_Commu": features_embeddings + features_commu_inferee,
        "Structure + Inferred_Commu": features_structure + features_commu_inferee,
        "Structure + Embeddings": features_structure + features_embeddings,
        "Full_features": features_structure + features_commu_inferee + features_embeddings
    }
    
    all_results = []
    G_name = "reel_Airports"
    G_obj = load_graphml_safe("graph_library/reel_Airports.graphml")
    
    #for i in [0.00,1.00]:
    for i in np.arange(0.00, 1.25, 0.25):
        sbm_val = f"{i:.2f}"
        pos_val = f"{1-i:.2f}"
        G_name = f"artificial_graph_sbmv2_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
        G_name_graph = f"artificial_graph_sbmv2_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
        
        # 1. Chargement des données d'entraînement
        G_train, dataset_train, dataset_eval, _, _, _ = gp.load_all_data_for_graph(G_name)
        G_obj = load_json(f"graph_library/{G_name_graph}.json")
    
        # 2. Injection de la vérité terrain
        #dataset_train = gp.enrich_dataset_with_ground_truth(dataset_train, G_obj, p_intra=0.6, q_inter=0.1)
        #dataset_eval = gp.enrich_dataset_with_ground_truth(dataset_eval, G_obj, p_intra=0.6, q_inter=0.1)
    
        pre_calculated_folds = gput._prepare_precalculated_folds(G_train, k=1)
        
        # 3. Lancement des tests
        for exp_name, feat_list in experiments.items():
            missing = set(feat_list) - set(dataset_train.columns)
            if missing:
                print(f" Exp {exp_name} : colonnes manquantes {missing}. Skip.")
                continue
    
            print(f" Running: {exp_name} for SBM={i}")
            Params, summary_optim = k_fold_cross_validation_exp(G_train, pre_calculated_folds, feat_list, n_trials=50)
            
            stats_df, model, X_test, _, y_test, _ = gp.train_and_test_xgboost(dataset_train, features=feat_list,
                                                                              parameters = Params, plot=False)
    
            importances = model.feature_importances_
            feat_imp_series = pd.Series(importances, index=feat_list).sort_values(ascending=False)
            print(f"Top 5 Feature Importance pour {exp_name} :")
            print(feat_imp_series.head(5).to_string())
    
            probs = model.predict_proba(X_test)[:, 1]
            preds = model.predict(X_test)
    
            if exp_name == "Ground_Truth_Only":
                model_GT, X_test_GT_only, y_test_GT_only, probs_GT, preds_GT_only = model, X_test, y_test, probs, preds
            elif exp_name == "Inferred_Commu only":
                model_s, X_test_s, y_test_s, probs_s, preds_s = model, X_test, y_test, probs, preds
                
            # Évaluation sur le dataset de référence FIXE (Graphe SBM 1.0)
            # On force l'ordre des colonnes de eval_dataset pour correspondre au modèle
            X_eval_fixed = dataset_eval[feat_list] 
            stats_eval_df = gp.get_performance_metrics(model, X_eval_fixed, dataset_eval["target"], "EXP_")
            
            all_results.append({
                "Ratio_SBM": i,
                "Experiment": exp_name,
                "AP_train": stats_df["Test_AP"].iloc[0],
                "AUC-ROC_train": stats_df["Test_AUC-ROC"].iloc[0],
                "AP_eval": stats_eval_df["EXP_AP"].iloc[0],
                "AUC-ROC_eval": stats_eval_df["EXP_AUC-ROC"].iloc[0],
                "Top_Feature": feat_imp_series.index[0], # On stocke la #1 pour analyse
                "Top_Importance": feat_imp_series.iloc[0]
            })
    
    # Recap final
    df_compare = pd.DataFrame(all_results)
    
    print("\n" + "="*50)
    print("RÉCAPITULATIF DES EXPÉRIENCES (SBM vs Ground Truth)")
    print("="*50)
    
    # On affiche sans l'index pour plus de clarté si tu as déjà une colonne 'exp_name'
    print(df_compare.to_string(index=False, float_format=lambda x: "{:,.4f}".format(x)))
    
    print("="*50 + "\n")
    
    output_dir = "outputs/results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "comparaison_finale_structure_vs_gt_v2_20260311.csv")
    df_compare.to_csv(output_path, index=False)
    print(f" Succès ! Fichier sauvegardé dans : {output_path}")
        
        