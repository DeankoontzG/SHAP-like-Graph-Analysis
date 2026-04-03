%load_ext autoreload
%autoreload 2

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
%load_ext autoreload
%autoreload 2

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
features_GT_proba = ['GT_proba']
features_GT_proba_et_sbm = ['GT_proba','GT_sbm_density', 'same_GT_sbm']
features_GT_sbm = ['GT_sbm_density', 'same_GT_sbm']
features_GT_pos = ['GT_pos_dist', 'GT_pos_dist_sq', 'GT_pos_had_mean', 'GT_pos_had_std', 'GT_pos_cos', 'GT_pos_rank']
features_GT = ['GT_sbm_density', 'same_GT_sbm', 'GT_pos_dist', 'GT_pos_dist_sq', 'GT_pos_had_mean', 'GT_pos_had_std', 'GT_pos_cos', 'GT_pos_rank']
features_structure = ['pr_u', 'pr_v','ppr_u', 'ppr_v', 'lcc_u', 'lcc_v', 'and_u', 'and_v', 'dc_u', 'dc_v', 'katz_u', 'katz_v',
                              'sp', 'jc', 'aa', 'cn','pa', 'ra']
features_commu_inferee = ['louvain_density', 'same_louvain', 'infomap_density', 'same_infomap', 'sbm_density', 'same_sbm',
                            'leiden_density', 'same_leiden', "surprise_density", "same_surprise", 
                            "significance_density", "same_significance"]
features_commu_surprise = ["surprise_density", "same_surprise"]
features_commu_infomap = ['infomap_density', 'same_infomap']
features_commu_sbm = ['sbm_density', 'same_sbm']
features_embeddings = ["deepwalk_cos", "deepwalk_rank", "deepwalk_had_mean",  "deepwalk_had_std",
                        "n2v_homophily_cos", "n2v_homophily_rank", "n2v_homophily_had_mean",  "n2v_homophily_had_std",
                      "crosswalk_cos", "crosswalk_rank", "crosswalk_had_mean",  "crosswalk_had_std"]

experiments = {
    "GT_absolue (proba + sbm)": features_GT_proba_et_sbm,
    "GT_absolue (proba)": features_GT_proba,
    "GT_sbm": features_GT_sbm,
    "GT_pos": features_GT_pos,
    "GT": features_GT,
    #"Structure_Only": features_structure,
    #"Embeddings_Only": features_embeddings,
    #"Significance" : features_commu_significance,
    #"Infomap" : features_commu_infomap,
    #"Surprise" : features_commu_surprise,
    "Sbm": features_commu_sbm,
    "Inferred_Commu only": features_commu_inferee,
    #"Embeddings + Inferred_Commu": features_embeddings + features_commu_inferee,
    #"Structure + Inferred_Commu": features_structure + features_commu_inferee,
    #"Structure + Embeddings": features_structure + features_embeddings,
    "Full_features": features_structure + features_commu_inferee + features_embeddings
}

all_results = []

#for i in [0.00,1.00]:
for i in np.arange(1.00, 1.25, 0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv4_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}_GT"
    G_name_obj = f"artificial_graph_sbmv4_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    # 1. Chargement des données d'entraînement
    G_train, dataset_train, dataset_eval, _, _, _ = gp.load_all_data_for_graph(G_name)
    G = load_graphml_safe(f"graph_library/{G_name_obj}.graphml")

    # --- VÉRIFICATION CIBLÉE SUR DATASET_EVAL ---
    print("\n" + "="*60)
    print("🔍 AUDIT DES COLONNES : GT_sbm & GT_proba")
    print("="*60)
    
    # 1. Analyse statistique des deux colonnes cibles
    for col in ['GT_sbm_density', 'GT_proba']:
        if col in dataset_eval.columns:
            series = dataset_eval[col]
            print(f"Analyse de {col}:")
            print(f"  - Type de donnée : {series.dtype}")
            print(f"  - Valeurs nulles : {series.isna().sum()}")
            print(f"  - Moyenne       : {series.mean():.6f}")
            print(f"  - Min / Max     : {series.min():.6f} / {series.max():.6f}")
            print(f"  - Valeurs uniques: {len(series.unique())}")
        else:
            print(f"❌ ERREUR : La colonne '{col}' est introuvable dans dataset_eval.")
        print("-" * 40)
    
    # 2. Affichage des 5 premières valeurs (Ta demande)
    print("👀 APERÇU DES 5 PREMIÈRES LIGNES :")
    if set(['GT_sbm', 'GT_proba']).issubset(dataset_eval.columns):
        print(dataset_eval[['GT_sbm', 'GT_proba']].head(5))
    else:
        # Affiche ce qui est disponible au cas où l'une manque
        cols_presentes = [c for c in ['GT_sbm', 'GT_proba'] if c in dataset_eval.columns]
        print(dataset_eval[cols_presentes].head(5))
    
    print("="*60)

    if 'P_matrix' in G.graph :        # Cas où on souhaite injecter la proba GT dans le graphe
        matrix_list = json.loads(G.graph['P_matrix'])
        P_matrix = np.array(matrix_list, dtype=float)        
        print(f"Matrice P récupérée avec succès pour la GT (Format: {P_matrix.shape})")
    else :
        print(f"Pas de P_Matrix à récupérer")
        P_matrix = None

    pre_calculated_folds = gput._prepare_precalculated_folds(G_train, k=1, P_matrix=P_matrix)
    
    # 3. Lancement des tests
    for exp_name, feat_list in experiments.items():
        missing = set(feat_list) - set(dataset_train.columns)
        if missing:
            print(f" Exp {exp_name} : colonnes manquantes {missing}. Skip.")
            print(set(dataset_train.columns))
            continue

        print(f" Running: {exp_name} for SBM={i}")
        Params, summary_optim = k_fold_cross_validation_exp(G_train, pre_calculated_folds, feat_list, n_trials=30)
        
        stats_df, model, X_test, _, y_test, _ = gp.train_and_test_xgboost(dataset_train, features=feat_list,
                                                                          parameters = Params, plot=False)

        importances = model.feature_importances_
        feat_imp_series = pd.Series(importances, index=feat_list).sort_values(ascending=False)
        print(f"Top 5 Feature Importance pour {exp_name} :")
        print(feat_imp_series.head(5).to_string())

        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

        if exp_name == "GT_sbm":
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
output_path = os.path.join(output_dir, "comparaison_finale_structure_shuffled_v4_20260401.csv")
df_compare.to_csv(output_path, index=False)
print(f" Succès ! Fichier sauvegardé dans : {output_path}")
features_GT_proba = ['GT_proba']
features_GT_proba_et_sbm = ['GT_proba','GT_sbm_density', 'same_GT_sbm']
features_GT_sbm = ['GT_sbm_density', 'same_GT_sbm']
features_GT_pos = ['GT_pos_dist', 'GT_pos_dist_sq', 'GT_pos_had_mean', 'GT_pos_had_std', 'GT_pos_cos', 'GT_pos_rank']
features_GT = ['GT_sbm_density', 'same_GT_sbm', 'GT_pos_dist', 'GT_pos_dist_sq', 'GT_pos_had_mean', 'GT_pos_had_std', 'GT_pos_cos', 'GT_pos_rank']
features_structure = ['pr_u', 'pr_v','ppr_u', 'ppr_v', 'lcc_u', 'lcc_v', 'and_u', 'and_v', 'dc_u', 'dc_v', 'katz_u', 'katz_v',
                              'sp', 'jc', 'aa', 'cn','pa', 'ra']
features_commu_inferee = ['louvain_density', 'same_louvain', 'infomap_density', 'same_infomap', 'sbm_density', 'same_sbm',
                            'leiden_density', 'same_leiden', "surprise_density", "same_surprise", 
                            "significance_density", "same_significance"]
features_commu_surprise = ["surprise_density", "same_surprise"]
features_commu_infomap = ['infomap_density', 'same_infomap']
features_commu_sbm = ['sbm_density', 'same_sbm']
features_embeddings = ["deepwalk_cos", "deepwalk_rank", "deepwalk_had_mean",  "deepwalk_had_std",
                        "n2v_homophily_cos", "n2v_homophily_rank", "n2v_homophily_had_mean",  "n2v_homophily_had_std",
                      "crosswalk_cos", "crosswalk_rank", "crosswalk_had_mean",  "crosswalk_had_std"]

experiments = {
    "GT_absolue (proba + sbm)": features_GT_proba_et_sbm,
    "GT_absolue (proba)": features_GT_proba,
    "GT_sbm": features_GT_sbm,
    "GT_pos": features_GT_pos,
    "GT": features_GT,
    #"Structure_Only": features_structure,
    #"Embeddings_Only": features_embeddings,
    #"Significance" : features_commu_significance,
    #"Infomap" : features_commu_infomap,
    #"Surprise" : features_commu_surprise,
    "Sbm": features_commu_sbm,
    "Inferred_Commu only": features_commu_inferee,
    #"Embeddings + Inferred_Commu": features_embeddings + features_commu_inferee,
    #"Structure + Inferred_Commu": features_structure + features_commu_inferee,
    #"Structure + Embeddings": features_structure + features_embeddings,
    "Full_features": features_structure + features_commu_inferee + features_embeddings
}

all_results = []

#for i in [0.00,1.00]:
for i in np.arange(1.00, 1.25, 0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv4_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}_GT"
    G_name_obj = f"artificial_graph_sbmv4_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    # 1. Chargement des données d'entraînement
    G_train, dataset_train, dataset_eval, _, _, _ = gp.load_all_data_for_graph(G_name)
    G = load_graphml_safe(f"graph_library/{G_name_obj}.graphml")

    # --- VÉRIFICATION CIBLÉE SUR DATASET_EVAL ---
    print("\n" + "="*60)
    print("🔍 AUDIT DES COLONNES : GT_sbm & GT_proba")
    print("="*60)
    
    # 1. Analyse statistique des deux colonnes cibles
    for col in ['GT_sbm_density', 'GT_proba']:
        if col in dataset_eval.columns:
            series = dataset_eval[col]
            print(f"Analyse de {col}:")
            print(f"  - Type de donnée : {series.dtype}")
            print(f"  - Valeurs nulles : {series.isna().sum()}")
            print(f"  - Moyenne       : {series.mean():.6f}")
            print(f"  - Min / Max     : {series.min():.6f} / {series.max():.6f}")
            print(f"  - Valeurs uniques: {len(series.unique())}")
        else:
            print(f"❌ ERREUR : La colonne '{col}' est introuvable dans dataset_eval.")
        print("-" * 40)
    
    # 2. Affichage des 5 premières valeurs (Ta demande)
    print("👀 APERÇU DES 5 PREMIÈRES LIGNES :")
    if set(['GT_sbm', 'GT_proba']).issubset(dataset_eval.columns):
        print(dataset_eval[['GT_sbm', 'GT_proba']].head(5))
    else:
        # Affiche ce qui est disponible au cas où l'une manque
        cols_presentes = [c for c in ['GT_sbm_density', 'GT_proba'] if c in dataset_eval.columns]
        print(dataset_eval[cols_presentes].head(5))
    
    print("="*60)

    if 'P_matrix' in G.graph :        # Cas où on souhaite injecter la proba GT dans le graphe
        matrix_list = json.loads(G.graph['P_matrix'])
        P_matrix = np.array(matrix_list, dtype=float)        
        print(f"Matrice P récupérée avec succès pour la GT (Format: {P_matrix.shape})")
    else :
        print(f"Pas de P_Matrix à récupérer")
        P_matrix = None

    pre_calculated_folds = gput._prepare_precalculated_folds(G_train, k=1, P_matrix=P_matrix)
    
    # 3. Lancement des tests
    for exp_name, feat_list in experiments.items():
        missing = set(feat_list) - set(dataset_train.columns)
        if missing:
            print(f" Exp {exp_name} : colonnes manquantes {missing}. Skip.")
            print(set(dataset_train.columns))
            continue

        print(f" Running: {exp_name} for SBM={i}")
        Params, summary_optim = k_fold_cross_validation_exp(G_train, pre_calculated_folds, feat_list, n_trials=30)
        
        stats_df, model, X_test, _, y_test, _ = gp.train_and_test_xgboost(dataset_train, features=feat_list,
                                                                          parameters = Params, plot=False)

        importances = model.feature_importances_
        feat_imp_series = pd.Series(importances, index=feat_list).sort_values(ascending=False)
        print(f"Top 5 Feature Importance pour {exp_name} :")
        print(feat_imp_series.head(5).to_string())

        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

        if exp_name == "GT_sbm":
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
output_path = os.path.join(output_dir, "comparaison_finale_structure_shuffled_v4_20260401.csv")
df_compare.to_csv(output_path, index=False)
print(f" Succès ! Fichier sauvegardé dans : {output_path}")
features_GT_proba = ['GT_proba']
features_GT_proba_et_sbm = ['GT_proba','GT_sbm_density', 'same_GT_sbm']
features_GT_sbm = ['GT_sbm_density', 'same_GT_sbm']
features_GT_pos = ['GT_pos_dist', 'GT_pos_dist_sq', 'GT_pos_had_mean', 'GT_pos_had_std', 'GT_pos_cos', 'GT_pos_rank']
features_GT = ['GT_sbm_density', 'same_GT_sbm', 'GT_pos_dist', 'GT_pos_dist_sq', 'GT_pos_had_mean', 'GT_pos_had_std', 'GT_pos_cos', 'GT_pos_rank']
features_structure = ['pr_u', 'pr_v','ppr_u', 'ppr_v', 'lcc_u', 'lcc_v', 'and_u', 'and_v', 'dc_u', 'dc_v', 'katz_u', 'katz_v',
                              'sp', 'jc', 'aa', 'cn','pa', 'ra']
features_commu_inferee = ['louvain_density', 'same_louvain', 'infomap_density', 'same_infomap', 'sbm_density', 'same_sbm',
                            'leiden_density', 'same_leiden', "surprise_density", "same_surprise", 
                            "significance_density", "same_significance"]
features_commu_surprise = ["surprise_density", "same_surprise"]
features_commu_infomap = ['infomap_density', 'same_infomap']
features_commu_sbm = ['sbm_density', 'same_sbm']
features_embeddings = ["deepwalk_cos", "deepwalk_rank", "deepwalk_had_mean",  "deepwalk_had_std",
                        "n2v_homophily_cos", "n2v_homophily_rank", "n2v_homophily_had_mean",  "n2v_homophily_had_std",
                      "crosswalk_cos", "crosswalk_rank", "crosswalk_had_mean",  "crosswalk_had_std"]

experiments = {
    "GT_absolue (proba + sbm)": features_GT_proba_et_sbm,
    "GT_absolue (proba)": features_GT_proba,
    "GT_sbm": features_GT_sbm,
    "GT_pos": features_GT_pos,
    "GT": features_GT,
    #"Structure_Only": features_structure,
    #"Embeddings_Only": features_embeddings,
    #"Significance" : features_commu_significance,
    #"Infomap" : features_commu_infomap,
    #"Surprise" : features_commu_surprise,
    "Sbm": features_commu_sbm,
    "Inferred_Commu only": features_commu_inferee,
    #"Embeddings + Inferred_Commu": features_embeddings + features_commu_inferee,
    #"Structure + Inferred_Commu": features_structure + features_commu_inferee,
    #"Structure + Embeddings": features_structure + features_embeddings,
    "Full_features": features_structure + features_commu_inferee + features_embeddings
}

all_results = []

#for i in [0.00,1.00]:
for i in np.arange(1.00, 1.25, 0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv4_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}_GT"
    G_name_obj = f"artificial_graph_sbmv4_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    # 1. Chargement des données d'entraînement
    G_train, dataset_train, dataset_eval, _, _, _ = gp.load_all_data_for_graph(G_name)
    G = load_graphml_safe(f"graph_library/{G_name_obj}.graphml")

    import json
    
    # 1. Chargement des données brutes depuis le graphe
    raw_data = json.loads(G.graph['GT_true_probs'])
    
    # 2. Extraction des valeurs (les probabilités)
    all_probs = list(raw_data.values())
    
    # 3. Calcul des statistiques
    distinct_probs = set(all_probs)
    n_distinct = len(distinct_probs)
    total_pairs = len(all_probs)
    
    print(f"--- ANALYSE DE GT_true_probs ---")
    print(f"Nombre total de paires de blocs enregistrées : {total_pairs}")
    print(f"Nombre de valeurs de probabilité DISTINCTES  : {n_distinct}")
    
    # 4. Petit aperçu des valeurs pour voir si elles sont crédibles
    print(f"Exemples de valeurs distinctes (Top 5) : {sorted(list(distinct_probs))[:5]}")
    
    # 5. Vérification rapide : est-ce qu'on a bien 15 communautés ?
    # (On déduit le nombre de blocs à partir des clés '0-0', '0-1', etc.)
    block_ids = set()
    for key in raw_data.keys():
        b1, b2 = map(int, key.split('-'))
        block_ids.update([b1, b2])
    
    print(f"Nombre de blocs détectés dans les clés : {len(block_ids)}")

    # --- VÉRIFICATION CIBLÉE SUR DATASET_EVAL ---
    print("\n" + "="*60)
    print("🔍 AUDIT DES COLONNES : GT_sbm & GT_proba")
    print("="*60)
    
    # 1. Analyse statistique des deux colonnes cibles
    for col in ['GT_sbm_density', 'GT_proba']:
        if col in dataset_eval.columns:
            series = dataset_eval[col]
            print(f"Analyse de {col}:")
            print(f"  - Type de donnée : {series.dtype}")
            print(f"  - Valeurs nulles : {series.isna().sum()}")
            print(f"  - Moyenne       : {series.mean():.6f}")
            print(f"  - Min / Max     : {series.min():.6f} / {series.max():.6f}")
            print(f"  - Valeurs uniques: {len(series.unique())}")
        else:
            print(f"❌ ERREUR : La colonne '{col}' est introuvable dans dataset_eval.")
        print("-" * 40)
    
    # 2. Affichage des 5 premières valeurs (Ta demande)
    print("👀 APERÇU DES 5 PREMIÈRES LIGNES :")
    if set(['GT_sbm', 'GT_proba']).issubset(dataset_eval.columns):
        print(dataset_eval[['GT_sbm', 'GT_proba']].head(5))
    else:
        # Affiche ce qui est disponible au cas où l'une manque
        cols_presentes = [c for c in ['GT_sbm_density', 'GT_proba'] if c in dataset_eval.columns]
        print(dataset_eval[cols_presentes].head(5))
    
    print("="*60)

    if 'P_matrix' in G.graph :        # Cas où on souhaite injecter la proba GT dans le graphe
        matrix_list = json.loads(G.graph['P_matrix'])
        P_matrix = np.array(matrix_list, dtype=float)        
        print(f"Matrice P récupérée avec succès pour la GT (Format: {P_matrix.shape})")
    else :
        print(f"Pas de P_Matrix à récupérer")
        P_matrix = None

    pre_calculated_folds = gput._prepare_precalculated_folds(G_train, k=1, P_matrix=P_matrix)
    
    # 3. Lancement des tests
    for exp_name, feat_list in experiments.items():
        missing = set(feat_list) - set(dataset_train.columns)
        if missing:
            print(f" Exp {exp_name} : colonnes manquantes {missing}. Skip.")
            print(set(dataset_train.columns))
            continue

        print(f" Running: {exp_name} for SBM={i}")
        Params, summary_optim = k_fold_cross_validation_exp(G_train, pre_calculated_folds, feat_list, n_trials=30)
        
        stats_df, model, X_test, _, y_test, _ = gp.train_and_test_xgboost(dataset_train, features=feat_list,
                                                                          parameters = Params, plot=False)

        importances = model.feature_importances_
        feat_imp_series = pd.Series(importances, index=feat_list).sort_values(ascending=False)
        print(f"Top 5 Feature Importance pour {exp_name} :")
        print(feat_imp_series.head(5).to_string())

        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

        if exp_name == "GT_sbm":
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
output_path = os.path.join(output_dir, "comparaison_finale_structure_shuffled_v4_20260401.csv")
df_compare.to_csv(output_path, index=False)
print(f" Succès ! Fichier sauvegardé dans : {output_path}")
features_GT_proba = ['GT_proba']
features_GT_proba_et_sbm = ['GT_proba','GT_sbm_density', 'same_GT_sbm']
features_GT_sbm = ['GT_sbm_density', 'same_GT_sbm']
features_GT_pos = ['GT_pos_dist', 'GT_pos_dist_sq', 'GT_pos_had_mean', 'GT_pos_had_std', 'GT_pos_cos', 'GT_pos_rank']
features_GT = ['GT_sbm_density', 'same_GT_sbm', 'GT_pos_dist', 'GT_pos_dist_sq', 'GT_pos_had_mean', 'GT_pos_had_std', 'GT_pos_cos', 'GT_pos_rank']
features_structure = ['pr_u', 'pr_v','ppr_u', 'ppr_v', 'lcc_u', 'lcc_v', 'and_u', 'and_v', 'dc_u', 'dc_v', 'katz_u', 'katz_v',
                              'sp', 'jc', 'aa', 'cn','pa', 'ra']
features_commu_inferee = ['louvain_density', 'same_louvain', 'infomap_density', 'same_infomap', 'sbm_density', 'same_sbm',
                            'leiden_density', 'same_leiden', "surprise_density", "same_surprise", 
                            "significance_density", "same_significance"]
features_commu_surprise = ["surprise_density", "same_surprise"]
features_commu_infomap = ['infomap_density', 'same_infomap']
features_commu_sbm = ['sbm_density', 'same_sbm']
features_embeddings = ["deepwalk_cos", "deepwalk_rank", "deepwalk_had_mean",  "deepwalk_had_std",
                        "n2v_homophily_cos", "n2v_homophily_rank", "n2v_homophily_had_mean",  "n2v_homophily_had_std",
                      "crosswalk_cos", "crosswalk_rank", "crosswalk_had_mean",  "crosswalk_had_std"]

experiments = {
    "GT_absolue (proba + sbm)": features_GT_proba_et_sbm,
    "GT_absolue (proba)": features_GT_proba,
    "GT_sbm": features_GT_sbm,
    "GT_pos": features_GT_pos,
    "GT": features_GT,
    #"Structure_Only": features_structure,
    #"Embeddings_Only": features_embeddings,
    #"Significance" : features_commu_significance,
    #"Infomap" : features_commu_infomap,
    #"Surprise" : features_commu_surprise,
    "Sbm": features_commu_sbm,
    "Inferred_Commu only": features_commu_inferee,
    #"Embeddings + Inferred_Commu": features_embeddings + features_commu_inferee,
    #"Structure + Inferred_Commu": features_structure + features_commu_inferee,
    #"Structure + Embeddings": features_structure + features_embeddings,
    "Full_features": features_structure + features_commu_inferee + features_embeddings
}

all_results = []

#for i in [0.00,1.00]:
for i in np.arange(1.00, 1.25, 0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv4_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}_GT"
    G_name_obj = f"artificial_graph_sbmv4_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    # 1. Chargement des données d'entraînement
    G_train, dataset_train, dataset_eval, _, _, _ = gp.load_all_data_for_graph(G_name)
    G = load_graphml_safe(f"graph_library/{G_name_obj}.graphml")

    # --- VÉRIFICATION CIBLÉE SUR DATASET_EVAL ---
    print("\n" + "="*60)
    print("🔍 AUDIT DES COLONNES : GT_sbm & GT_proba")
    print("="*60)
    
    # 1. Analyse statistique des deux colonnes cibles
    for col in ['GT_sbm_density', 'GT_proba']:
        if col in dataset_eval.columns:
            series = dataset_eval[col]
            print(f"Analyse de {col}:")
            print(f"  - Type de donnée : {series.dtype}")
            print(f"  - Valeurs nulles : {series.isna().sum()}")
            print(f"  - Moyenne       : {series.mean():.6f}")
            print(f"  - Min / Max     : {series.min():.6f} / {series.max():.6f}")
            print(f"  - Valeurs uniques: {len(series.unique())}")
        else:
            print(f"❌ ERREUR : La colonne '{col}' est introuvable dans dataset_eval.")
        print("-" * 40)
    
    # 2. Affichage des 5 premières valeurs (Ta demande)
    print("👀 APERÇU DES 5 PREMIÈRES LIGNES :")
    if set(['GT_sbm', 'GT_proba']).issubset(dataset_eval.columns):
        print(dataset_eval[['GT_sbm', 'GT_proba']].head(5))
    else:
        # Affiche ce qui est disponible au cas où l'une manque
        cols_presentes = [c for c in ['GT_sbm_density', 'GT_proba'] if c in dataset_eval.columns]
        print(dataset_eval[cols_presentes].head(5))
    
    print("="*60)

    if 'P_matrix' in G.graph :        # Cas où on souhaite injecter la proba GT dans le graphe
        matrix_list = json.loads(G.graph['P_matrix'])
        P_matrix = np.array(matrix_list, dtype=float)        
        print(f"Matrice P récupérée avec succès pour la GT (Format: {P_matrix.shape})")
    else :
        print(f"Pas de P_Matrix à récupérer")
        P_matrix = None

    pre_calculated_folds = gput._prepare_precalculated_folds(G_train, k=1, P_matrix=P_matrix)
    
    # 3. Lancement des tests
    for exp_name, feat_list in experiments.items():
        missing = set(feat_list) - set(dataset_train.columns)
        if missing:
            print(f" Exp {exp_name} : colonnes manquantes {missing}. Skip.")
            print(set(dataset_train.columns))
            continue

        print(f" Running: {exp_name} for SBM={i}")
        Params, summary_optim = k_fold_cross_validation_exp(G_train, pre_calculated_folds, feat_list, n_trials=30)
        
        stats_df, model, X_test, _, y_test, _ = gp.train_and_test_xgboost(dataset_train, features=feat_list,
                                                                          parameters = Params, plot=False)

        importances = model.feature_importances_
        feat_imp_series = pd.Series(importances, index=feat_list).sort_values(ascending=False)
        print(f"Top 5 Feature Importance pour {exp_name} :")
        print(feat_imp_series.head(5).to_string())

        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

        if exp_name == "GT_sbm":
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
output_path = os.path.join(output_dir, "comparaison_finale_structure_shuffled_v4_20260401.csv")
df_compare.to_csv(output_path, index=False)
print(f" Succès ! Fichier sauvegardé dans : {output_path}")
%load_ext autoreload
%autoreload 2

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
features_GT_proba = ['GT_proba']
features_GT_proba_et_sbm = ['GT_proba','GT_sbm_density', 'same_GT_sbm']
features_GT_sbm = ['GT_sbm_density', 'same_GT_sbm']
features_GT_pos = ['GT_pos_dist', 'GT_pos_dist_sq', 'GT_pos_had_mean', 'GT_pos_had_std', 'GT_pos_cos', 'GT_pos_rank']
features_GT = ['GT_sbm_density', 'same_GT_sbm', 'GT_pos_dist', 'GT_pos_dist_sq', 'GT_pos_had_mean', 'GT_pos_had_std', 'GT_pos_cos', 'GT_pos_rank']
features_structure = ['pr_u', 'pr_v','ppr_u', 'ppr_v', 'lcc_u', 'lcc_v', 'and_u', 'and_v', 'dc_u', 'dc_v', 'katz_u', 'katz_v',
                              'sp', 'jc', 'aa', 'cn','pa', 'ra']
features_commu_inferee = ['louvain_density', 'same_louvain', 'infomap_density', 'same_infomap', 'sbm_density', 'same_sbm',
                            'leiden_density', 'same_leiden', "surprise_density", "same_surprise", 
                            "significance_density", "same_significance"]
features_commu_surprise = ["surprise_density", "same_surprise"]
features_commu_infomap = ['infomap_density', 'same_infomap']
features_commu_sbm = ['sbm_density', 'same_sbm']
features_embeddings = ["deepwalk_cos", "deepwalk_rank", "deepwalk_had_mean",  "deepwalk_had_std",
                        "n2v_homophily_cos", "n2v_homophily_rank", "n2v_homophily_had_mean",  "n2v_homophily_had_std",
                      "crosswalk_cos", "crosswalk_rank", "crosswalk_had_mean",  "crosswalk_had_std"]

experiments = {
    "GT_absolue (proba + sbm)": features_GT_proba_et_sbm,
    "GT_absolue (proba)": features_GT_proba,
    "GT_sbm": features_GT_sbm,
    "GT_pos": features_GT_pos,
    "GT": features_GT,
    #"Structure_Only": features_structure,
    #"Embeddings_Only": features_embeddings,
    #"Significance" : features_commu_significance,
    #"Infomap" : features_commu_infomap,
    #"Surprise" : features_commu_surprise,
    "Sbm": features_commu_sbm,
    "Inferred_Commu only": features_commu_inferee,
    #"Embeddings + Inferred_Commu": features_embeddings + features_commu_inferee,
    #"Structure + Inferred_Commu": features_structure + features_commu_inferee,
    #"Structure + Embeddings": features_structure + features_embeddings,
    "Full_features": features_structure + features_commu_inferee + features_embeddings
}

all_results = []

#for i in [0.00,1.00]:
for i in np.arange(1.00, 1.25, 0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv4_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}_GT"
    G_name_obj = f"artificial_graph_sbmv4_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    # 1. Chargement des données d'entraînement
    G_train, dataset_train, dataset_eval, _, _, _ = gp.load_all_data_for_graph(G_name)
    G = load_graphml_safe(f"graph_library/{G_name_obj}.graphml")

    # --- VÉRIFICATION CIBLÉE SUR DATASET_EVAL ---
    print("\n" + "="*60)
    print("🔍 AUDIT DES COLONNES : GT_sbm & GT_proba")
    print("="*60)
    
    # 1. Analyse statistique des deux colonnes cibles
    for col in ['GT_sbm_density', 'GT_proba']:
        if col in dataset_eval.columns:
            series = dataset_eval[col]
            print(f"Analyse de {col}:")
            print(f"  - Type de donnée : {series.dtype}")
            print(f"  - Valeurs nulles : {series.isna().sum()}")
            print(f"  - Moyenne       : {series.mean():.6f}")
            print(f"  - Min / Max     : {series.min():.6f} / {series.max():.6f}")
            print(f"  - Valeurs uniques: {len(series.unique())}")
        else:
            print(f"❌ ERREUR : La colonne '{col}' est introuvable dans dataset_eval.")
        print("-" * 40)
    
    # 2. Affichage des 5 premières valeurs (Ta demande)
    print("👀 APERÇU DES 5 PREMIÈRES LIGNES :")
    if set(['GT_sbm', 'GT_proba']).issubset(dataset_eval.columns):
        print(dataset_eval[['GT_sbm', 'GT_proba']].head(5))
    else:
        # Affiche ce qui est disponible au cas où l'une manque
        cols_presentes = [c for c in ['GT_sbm_density', 'GT_proba'] if c in dataset_eval.columns]
        print(dataset_eval[cols_presentes].head(5))
    
    print("="*60)

    if 'P_matrix' in G.graph :        # Cas où on souhaite injecter la proba GT dans le graphe
        matrix_list = json.loads(G.graph['P_matrix'])
        P_matrix = np.array(matrix_list, dtype=float)        
        print(f"Matrice P récupérée avec succès pour la GT (Format: {P_matrix.shape})")
    else :
        print(f"Pas de P_Matrix à récupérer")
        P_matrix = None

    pre_calculated_folds = gput._prepare_precalculated_folds(G_train, k=1, P_matrix=P_matrix)
    
    # 3. Lancement des tests
    for exp_name, feat_list in experiments.items():
        missing = set(feat_list) - set(dataset_train.columns)
        if missing:
            print(f" Exp {exp_name} : colonnes manquantes {missing}. Skip.")
            print(set(dataset_train.columns))
            continue

        print(f" Running: {exp_name} for SBM={i}")
        Params, summary_optim = k_fold_cross_validation_exp(G_train, pre_calculated_folds, feat_list, n_trials=30)
        
        stats_df, model, X_test, _, y_test, _ = gp.train_and_test_xgboost(dataset_train, features=feat_list,
                                                                          parameters = Params, plot=False)

        importances = model.feature_importances_
        feat_imp_series = pd.Series(importances, index=feat_list).sort_values(ascending=False)
        print(f"Top 5 Feature Importance pour {exp_name} :")
        print(feat_imp_series.head(5).to_string())

        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

        if exp_name == "GT_sbm":
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
output_path = os.path.join(output_dir, "comparaison_finale_structure_shuffled_v4_20260401.csv")
df_compare.to_csv(output_path, index=False)
print(f" Succès ! Fichier sauvegardé dans : {output_path}")
features_GT_proba = ['GT_proba']
features_GT_sbm = ['GT_sbm_density', 'GT_degrees_sbm_u', 'GT_degrees_sbm_v']
features_GT_pos = ['GT_pos_dist', 'GT_degrees_spatial_u','GT_degrees_spatial_v']
features_structure = ['pr_u', 'pr_v','ppr_u', 'ppr_v', 'lcc_u', 'lcc_v', 'and_u', 'and_v', 'dc_u', 'dc_v', 'katz_u', 'katz_v',
                              'sp', 'jc', 'aa', 'cn','pa', 'ra']
features_commu_inferee = ['louvain_density', 'same_louvain', 'infomap_density', 'same_infomap', 'sbm_density', 'same_sbm',
                            'leiden_density', 'same_leiden', "surprise_density", "same_surprise", 
                            "significance_density", "same_significance"]
features_commu_surprise = ["surprise_density", "same_surprise"]
features_commu_infomap = ['infomap_density', 'same_infomap']
features_commu_sbm = ['sbm_density', 'same_sbm']
features_embeddings = ["deepwalk_cos", "deepwalk_rank", "deepwalk_had_mean",  "deepwalk_had_std",
                        "n2v_homophily_cos", "n2v_homophily_rank", "n2v_homophily_had_mean",  "n2v_homophily_had_std",
                      "crosswalk_cos", "crosswalk_rank", "crosswalk_had_mean",  "crosswalk_had_std"]

experiments = {
    "GT_absolue (proba)": features_GT_proba,
    "GT_sbm": features_GT_sbm,
    "GT_pos": features_GT_pos,
    "GT": features_GT_sbm + features_GT_pos,
    "Structure_Only": features_structure,
    "Embeddings_Only": features_embeddings,
    "Inferred_Commu only": features_commu_inferee,
    "Embeddings + Inferred_Commu": features_embeddings + features_commu_inferee,
    "Structure + Inferred_Commu": features_structure + features_commu_inferee,
    "Structure + Embeddings": features_structure + features_embeddings,
    "Full_features": features_structure + features_commu_inferee + features_embeddings
}

all_results = []

#for i in [0.00,1.00]:
for i in np.arange(0.00, 1.25, 0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv4_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}_GT"
    G_name_obj = f"artificial_graph_sbmv4_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    # 1. Chargement des données d'entraînement
    G_train, dataset_train, dataset_eval, _, _, _ = gp.load_all_data_for_graph(G_name)
    G = load_graphml_safe(f"graph_library/{G_name_obj}.graphml")

    if 'P_matrix' in G.graph :        # Cas où on souhaite injecter la proba GT dans le graphe
        matrix_list = json.loads(G.graph['P_matrix'])
        P_matrix = np.array(matrix_list, dtype=float)        
        print(f"Matrice P récupérée avec succès pour la GT (Format: {P_matrix.shape})")
    else :
        print(f"Pas de P_Matrix à récupérer")
        P_matrix = None

    pre_calculated_folds = gput._prepare_precalculated_folds(G_train, k=1, P_matrix=P_matrix)
    
    # 3. Lancement des tests
    for exp_name, feat_list in experiments.items():
        missing = set(feat_list) - set(dataset_train.columns)
        if missing:
            print(f" Exp {exp_name} : colonnes manquantes {missing}. Skip.")
            print(set(dataset_train.columns))
            continue

        print(f" Running: {exp_name} for SBM={i}")
        Params, summary_optim = k_fold_cross_validation_exp(G_train, pre_calculated_folds, feat_list, n_trials=30)
        
        stats_df, model, X_test, _, y_test, _ = gp.train_and_test_xgboost(dataset_train, features=feat_list,
                                                                          parameters = Params, plot=False)

        importances = model.feature_importances_
        feat_imp_series = pd.Series(importances, index=feat_list).sort_values(ascending=False)
        print(f"Top 5 Feature Importance pour {exp_name} :")
        print(feat_imp_series.head(5).to_string())

        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

        if exp_name == "GT_sbm":
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
output_path = os.path.join(output_dir, "comparaison_finale_structure_shuffled_v4_20260401.csv")
df_compare.to_csv(output_path, index=False)
print(f" Succès ! Fichier sauvegardé dans : {output_path}")
%load_ext autoreload
%autoreload 2

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
features_GT_proba = ['GT_proba']
features_GT_sbm = ['GT_sbm_density', 'GT_degrees_sbm_u', 'GT_degrees_sbm_v']
features_GT_pos = ['GT_pos_dist', 'GT_degrees_spatial_u','GT_degrees_spatial_v']
features_structure = ['pr_u', 'pr_v','ppr_u', 'ppr_v', 'lcc_u', 'lcc_v', 'and_u', 'and_v', 'dc_u', 'dc_v', 'katz_u', 'katz_v',
                              'sp', 'jc', 'aa', 'cn','pa', 'ra']
features_commu_inferee = ['louvain_density', 'same_louvain', 'infomap_density', 'same_infomap', 'sbm_density', 'same_sbm',
                            'leiden_density', 'same_leiden', "surprise_density", "same_surprise", 
                            "significance_density", "same_significance"]
features_commu_surprise = ["surprise_density", "same_surprise"]
features_commu_infomap = ['infomap_density', 'same_infomap']
features_commu_sbm = ['sbm_density', 'same_sbm']
features_embeddings = ["deepwalk_cos", "deepwalk_rank", "deepwalk_had_mean",  "deepwalk_had_std",
                        "n2v_homophily_cos", "n2v_homophily_rank", "n2v_homophily_had_mean",  "n2v_homophily_had_std",
                      "crosswalk_cos", "crosswalk_rank", "crosswalk_had_mean",  "crosswalk_had_std"]

experiments = {
    "GT_absolue (proba)": features_GT_proba,
    "GT_sbm": features_GT_sbm,
    "GT_pos": features_GT_pos,
    "GT": features_GT_sbm + features_GT_pos,
    "Structure_Only": features_structure,
    "Embeddings_Only": features_embeddings,
    "Inferred_Commu only": features_commu_inferee,
    "Embeddings + Inferred_Commu": features_embeddings + features_commu_inferee,
    "Structure + Inferred_Commu": features_structure + features_commu_inferee,
    "Structure + Embeddings": features_structure + features_embeddings,
    "Full_features": features_structure + features_commu_inferee + features_embeddings
}

all_results = []

#for i in [0.00,1.00]:
for i in np.arange(0.00, 1.25, 0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv4_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}_GT"
    G_name_obj = f"artificial_graph_sbmv4_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    # 1. Chargement des données d'entraînement
    G_train, dataset_train, dataset_eval, _, _, _ = gp.load_all_data_for_graph(G_name)
    G = load_graphml_safe(f"graph_library/{G_name_obj}.graphml")

    if 'P_matrix' in G.graph :        # Cas où on souhaite injecter la proba GT dans le graphe
        matrix_list = json.loads(G.graph['P_matrix'])
        P_matrix = np.array(matrix_list, dtype=float)        
        print(f"Matrice P récupérée avec succès pour la GT (Format: {P_matrix.shape})")
    else :
        print(f"Pas de P_Matrix à récupérer")
        P_matrix = None

    pre_calculated_folds = gput._prepare_precalculated_folds(G_train, k=1, P_matrix=P_matrix)
    
    # 3. Lancement des tests
    for exp_name, feat_list in experiments.items():
        missing = set(feat_list) - set(dataset_train.columns)
        if missing:
            print(f" Exp {exp_name} : colonnes manquantes {missing}. Skip.")
            print(set(dataset_train.columns))
            continue

        print(f" Running: {exp_name} for SBM={i}")
        Params, summary_optim = k_fold_cross_validation_exp(G_train, pre_calculated_folds, feat_list, n_trials=30)
        
        stats_df, model, X_test, _, y_test, _ = gp.train_and_test_xgboost(dataset_train, features=feat_list,
                                                                          parameters = Params, plot=False)

        importances = model.feature_importances_
        feat_imp_series = pd.Series(importances, index=feat_list).sort_values(ascending=False)
        print(f"Top 5 Feature Importance pour {exp_name} :")
        print(feat_imp_series.head(5).to_string())

        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

        if exp_name == "GT_sbm":
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
output_path = os.path.join(output_dir, "comparaison_finale_structure_shuffled_v4_20260401.csv")
df_compare.to_csv(output_path, index=False)
print(f" Succès ! Fichier sauvegardé dans : {output_path}")
features_GT_proba = ['GT_proba']
features_GT_sbm = ['GT_sbm_density', 'GT_degrees_sbm_u', 'GT_degrees_sbm_v']
features_GT_pos = ['GT_pos_dist', 'GT_degrees_spatial_u','GT_degrees_spatial_v']
features_structure = ['pr_u', 'pr_v','ppr_u', 'ppr_v', 'lcc_u', 'lcc_v', 'and_u', 'and_v', 'dc_u', 'dc_v', 'katz_u', 'katz_v',
                              'sp', 'jc', 'aa', 'cn','pa', 'ra']
features_commu_inferee = ['louvain_density', 'same_louvain', 'infomap_density', 'same_infomap', 'sbm_density', 'same_sbm',
                            'leiden_density', 'same_leiden', "surprise_density", "same_surprise", 
                            "significance_density", "same_significance"]
features_commu_surprise = ["surprise_density", "same_surprise"]
features_commu_infomap = ['infomap_density', 'same_infomap']
features_commu_sbm = ['sbm_density', 'same_sbm']
features_embeddings = ["deepwalk_cos", "deepwalk_rank", "deepwalk_had_mean",  "deepwalk_had_std",
                        "n2v_homophily_cos", "n2v_homophily_rank", "n2v_homophily_had_mean",  "n2v_homophily_had_std",
                      "crosswalk_cos", "crosswalk_rank", "crosswalk_had_mean",  "crosswalk_had_std"]

experiments = {
    "GT_absolue (proba)": features_GT_proba,
    "GT_sbm": features_GT_sbm,
    "GT_pos": features_GT_pos,
    "GT": features_GT_sbm + features_GT_pos,
    "Structure_Only": features_structure,
    "Embeddings_Only": features_embeddings,
    "Inferred_Commu only": features_commu_inferee,
    "Embeddings + Inferred_Commu": features_embeddings + features_commu_inferee,
    "Structure + Inferred_Commu": features_structure + features_commu_inferee,
    "Structure + Embeddings": features_structure + features_embeddings,
    "Full_features": features_structure + features_commu_inferee + features_embeddings
}

all_results = []

#for i in [0.00,1.00]:
for i in np.arange(0.00, 1.25, 0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv4_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}_GT"
    G_name_obj = f"artificial_graph_sbmv4_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    # 1. Chargement des données d'entraînement
    G_train, dataset_train, dataset_eval, _, _, _ = gp.load_all_data_for_graph(G_name)
    G = load_graphml_safe(f"graph_library/{G_name_obj}.graphml")

    if 'GroundTruth_JSON' in G.graph:
        print(f"[INIT] Extraction de la GT GroundTruth pour {G_name}...")
        gt_raw = json.loads(G.graph['GroundTruth_JSON'])
        
        GT = {'GT_sbm_matrix': np.array(gt_raw['GT_sbm_matrix']),
            'GT_pos': np.array(gt_raw['GT_pos']),
            'GT_sbm_id': np.array(gt_raw['GT_sbm_id']),
            'GT_degrees_sbm': np.array(gt_raw['GT_degrees_sbm']),
            'GT_degrees_spatial': np.array(gt_raw['GT_degrees_spatial'])
             }
        
        if 'P_matrix_JSON' in G.graph:
            print("P_matrix trouvée.")
            GT['GT_proba'] = np.array(json.loads(G.graph['P_matrix_JSON']))
    else:
        print("[WARNING] Aucune GroundTruth_JSON trouvée dans G.graph")
        GT = None

    pre_calculated_folds = gput._prepare_precalculated_folds(G_train, k=1, GroundTruth=GT)
    
    # 3. Lancement des tests
    for exp_name, feat_list in experiments.items():
        missing = set(feat_list) - set(dataset_train.columns)
        if missing:
            print(f" Exp {exp_name} : colonnes manquantes {missing}. Skip.")
            print(set(dataset_train.columns))
            continue

        print(f" Running: {exp_name} for SBM={i}")
        Params, summary_optim = k_fold_cross_validation_exp(G_train, pre_calculated_folds, feat_list, n_trials=30)
        
        stats_df, model, X_test, _, y_test, _ = gp.train_and_test_xgboost(dataset_train, features=feat_list,
                                                                          parameters = Params, plot=False)

        importances = model.feature_importances_
        feat_imp_series = pd.Series(importances, index=feat_list).sort_values(ascending=False)
        print(f"Top 5 Feature Importance pour {exp_name} :")
        print(feat_imp_series.head(5).to_string())

        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

        if exp_name == "GT_sbm":
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
output_path = os.path.join(output_dir, "comparaison_finale_structure_shuffled_v4_20260401.csv")
df_compare.to_csv(output_path, index=False)
print(f" Succès ! Fichier sauvegardé dans : {output_path}")
features_GT_proba = ['GT_proba']
features_GT_sbm = ['GT_sbm_density', 'GT_degrees_sbm_u', 'GT_degrees_sbm_v']
features_GT_pos = ['GT_pos_dist', 'GT_degrees_spatial_u','GT_degrees_spatial_v']
features_structure = ['pr_u', 'pr_v','ppr_u', 'ppr_v', 'lcc_u', 'lcc_v', 'and_u', 'and_v', 'dc_u', 'dc_v', 'katz_u', 'katz_v',
                              'sp', 'jc', 'aa', 'cn','pa', 'ra']
features_commu_inferee = ['louvain_density', 'same_louvain', 'infomap_density', 'same_infomap', 'sbm_density', 'same_sbm',
                            'leiden_density', 'same_leiden', "surprise_density", "same_surprise", 
                            "significance_density", "same_significance"]
features_commu_surprise = ["surprise_density", "same_surprise"]
features_commu_infomap = ['infomap_density', 'same_infomap']
features_commu_sbm = ['sbm_density', 'same_sbm']
features_embeddings = ["deepwalk_cos", "deepwalk_rank", "deepwalk_had_mean",  "deepwalk_had_std",
                        "n2v_homophily_cos", "n2v_homophily_rank", "n2v_homophily_had_mean",  "n2v_homophily_had_std",
                      "crosswalk_cos", "crosswalk_rank", "crosswalk_had_mean",  "crosswalk_had_std"]

experiments = {
    "GT_absolue (proba)": features_GT_proba,
    "GT_sbm": features_GT_sbm,
    "GT_pos": features_GT_pos,
    "GT": features_GT_sbm + features_GT_pos,
    "Structure_Only": features_structure,
    "Embeddings_Only": features_embeddings,
    "Inferred_Commu only": features_commu_inferee,
    "Embeddings + Inferred_Commu": features_embeddings + features_commu_inferee,
    "Structure + Inferred_Commu": features_structure + features_commu_inferee,
    "Structure + Embeddings": features_structure + features_embeddings,
    "Full_features": features_structure + features_commu_inferee + features_embeddings
}

all_results = []

#for i in [0.00,1.00]:
for i in np.arange(1.00, -0.25, -0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv4_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}_GT"
    G_name_obj = f"artificial_graph_sbmv4_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    # 1. Chargement des données d'entraînement
    G_train, dataset_train, dataset_eval, _, _, _ = gp.load_all_data_for_graph(G_name)
    G = load_graphml_safe(f"graph_library/{G_name_obj}.graphml")

    if 'GroundTruth_JSON' in G.graph:
        print(f"[INIT] Extraction de la GT GroundTruth pour {G_name}...")
        gt_raw = json.loads(G.graph['GroundTruth_JSON'])
        
        GT = {'GT_sbm_matrix': np.array(gt_raw['GT_sbm_matrix']),
            'GT_pos': np.array(gt_raw['GT_pos']),
            'GT_sbm_id': np.array(gt_raw['GT_sbm_id']),
            'GT_degrees_sbm': np.array(gt_raw['GT_degrees_sbm']),
            'GT_degrees_spatial': np.array(gt_raw['GT_degrees_spatial'])
             }
        
        if 'P_matrix_JSON' in G.graph:
            print("P_matrix trouvée.")
            GT['GT_proba'] = np.array(json.loads(G.graph['P_matrix_JSON']))
    else:
        print("[WARNING] Aucune GroundTruth_JSON trouvée dans G.graph")
        GT = None

    pre_calculated_folds = gput._prepare_precalculated_folds(G_train, k=1, GroundTruth=GT)
    
    # 3. Lancement des tests
    for exp_name, feat_list in experiments.items():
        missing = set(feat_list) - set(dataset_train.columns)
        if missing:
            print(f" Exp {exp_name} : colonnes manquantes {missing}. Skip.")
            print(set(dataset_train.columns))
            continue

        print(f" Running: {exp_name} for SBM={i}")
        Params, summary_optim = k_fold_cross_validation_exp(G_train, pre_calculated_folds, feat_list, n_trials=30)
        
        stats_df, model, X_test, _, y_test, _ = gp.train_and_test_xgboost(dataset_train, features=feat_list,
                                                                          parameters = Params, plot=False)

        importances = model.feature_importances_
        feat_imp_series = pd.Series(importances, index=feat_list).sort_values(ascending=False)
        print(f"Top 5 Feature Importance pour {exp_name} :")
        print(feat_imp_series.head(5).to_string())

        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

        if exp_name == "GT_sbm":
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
output_path = os.path.join(output_dir, "comparaison_finale_structure_shuffled_v4_20260401.csv")
df_compare.to_csv(output_path, index=False)
print(f" Succès ! Fichier sauvegardé dans : {output_path}")
def plot_f1_comparison(df_results, metric="F1-Score", name ="OUBLI"):
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Création du lineplot
    # On utilise 'Experiment' pour différencier les lignes par couleur et style
    ax = sns.lineplot(
        data=df_results, 
        x="Ratio_SBM", 
        y= metric, 
        hue="Experiment", 
        style="Experiment", 
        markers=True, 
        dashes=False,
        linewidth=2.5,
        markersize=8
    )
    
    # Personnalisation des axes
    plt.title(f"Évolution du {metric} selon le ratio SBM vs POS", fontsize=15, pad=20)
    plt.xlabel("Ratio SBM (Structure de communauté forte → faible)", fontsize=12)
    plt.ylabel(f"{metric} (Link Prediction)", fontsize=12)
    
    # Inversion de l'axe X si tu veux montrer la difficulté croissante
    # plt.gca().invert_xaxis() 
    
    plt.legend(title="Configurations Features", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0.00, 1.00) # Pour bien voir l'échelle de 0 à 1
    
    plt.tight_layout()

    output_dir = os.path.join("outputs", "plots")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{name}_{metric}_per_SBM_ratio.png")
    
    # 3. La ligne magique
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"Graphique sauvegardé ici : {save_path}")
    
    plt.show()

df_compare = pd.read_csv("outputs/results/comparaison_finale_structure_shuffled_v4_20260401.csv")

print(df_compare.head(8))
plot_f1_comparison(df_compare.head(4), metric="AUC-ROC_eval", name = "artificial_graphs_shuffled_v4_20260401" )
def plot_f1_comparison(df_results, metric="F1-Score", name ="OUBLI"):
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Création du lineplot
    # On utilise 'Experiment' pour différencier les lignes par couleur et style
    ax = sns.lineplot(
        data=df_results, 
        x="Ratio_SBM", 
        y= metric, 
        hue="Experiment", 
        style="Experiment", 
        markers=True, 
        dashes=False,
        linewidth=2.5,
        markersize=8
    )
    
    # Personnalisation des axes
    plt.title(f"Évolution du {metric} selon le ratio SBM vs POS", fontsize=15, pad=20)
    plt.xlabel("Ratio SBM (Structure de communauté forte → faible)", fontsize=12)
    plt.ylabel(f"{metric} (Link Prediction)", fontsize=12)
    
    # Inversion de l'axe X si tu veux montrer la difficulté croissante
    # plt.gca().invert_xaxis() 
    
    plt.legend(title="Configurations Features", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0.00, 1.00) # Pour bien voir l'échelle de 0 à 1
    
    plt.tight_layout()

    output_dir = os.path.join("outputs", "plots")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{name}_{metric}_per_SBM_ratio.png")
    
    # 3. La ligne magique
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"Graphique sauvegardé ici : {save_path}")
    
    plt.show()

df_compare = pd.read_csv("outputs/results/comparaison_finale_structure_shuffled_v4_20260401.csv")

plot_f1_comparison(df_compare, metric="AUC-ROC_eval", name = "artificial_graphs_shuffled_v4_20260401" )
def plot_f1_comparison(df_results, metric="F1-Score", name ="OUBLI"):
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Création du lineplot
    # On utilise 'Experiment' pour différencier les lignes par couleur et style
    ax = sns.lineplot(
        data=df_results, 
        x="Ratio_SBM", 
        y= metric, 
        hue="Experiment", 
        style="Experiment", 
        markers=True, 
        dashes=False,
        linewidth=2.5,
        markersize=8
    )
    
    # Personnalisation des axes
    plt.title(f"Évolution du {metric} selon le ratio SBM vs POS", fontsize=15, pad=20)
    plt.xlabel("Ratio SBM (Structure de communauté forte → faible)", fontsize=12)
    plt.ylabel(f"{metric} (Link Prediction)", fontsize=12)
    
    # Inversion de l'axe X si tu veux montrer la difficulté croissante
    # plt.gca().invert_xaxis() 
    
    plt.legend(title="Configurations Features", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0.59, 0.95) # Pour bien voir l'échelle de 0 à 1
    
    plt.tight_layout()

    output_dir = os.path.join("outputs", "plots")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{name}_{metric}_per_SBM_ratio.png")
    
    # 3. La ligne magique
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"Graphique sauvegardé ici : {save_path}")
    
    plt.show()

df_compare = pd.read_csv("outputs/results/comparaison_finale_structure_shuffled_v4_20260401.csv")

plot_f1_comparison(df_compare, metric="AUC-ROC_eval", name = "artificial_graphs_shuffled_v4_20260401" )
features_GT_proba = ['GT_proba']
features_GT_sbm = ['GT_sbm_density', 'GT_degrees_sbm_u', 'GT_degrees_sbm_v']
features_GT_pos = ['GT_pos_dist', 'GT_degrees_spatial_u','GT_degrees_spatial_v']
features_structure = ['pr_u', 'pr_v','ppr_u', 'ppr_v', 'lcc_u', 'lcc_v', 'and_u', 'and_v', 'dc_u', 'dc_v', 'katz_u', 'katz_v',
                              'sp', 'jc', 'aa', 'cn','pa', 'ra']
features_commu_inferee = ['louvain_density', 'same_louvain', 'infomap_density', 'same_infomap', 'sbm_density', 'same_sbm',
                            'leiden_density', 'same_leiden', "surprise_density", "same_surprise", 
                            "significance_density", "same_significance"]
features_commu_surprise = ["surprise_density", "same_surprise"]
features_commu_infomap = ['infomap_density', 'same_infomap']
features_commu_sbm = ['sbm_density', 'same_sbm']
features_embeddings = ["deepwalk_cos", "deepwalk_rank", "deepwalk_had_mean",  "deepwalk_had_std",
                        "n2v_homophily_cos", "n2v_homophily_rank", "n2v_homophily_had_mean",  "n2v_homophily_had_std",
                      "crosswalk_cos", "crosswalk_rank", "crosswalk_had_mean",  "crosswalk_had_std"]

experiments = {
    #"GT_absolue (proba)": features_GT_proba,
    "GT_sbm": features_GT_sbm,
    "GT_pos": features_GT_pos,
    "GT": features_GT_sbm + features_GT_pos,
    "Structure_Only": features_structure,
    "Embeddings_Only": features_embeddings,
    "Inferred_Commu only": features_commu_inferee,
    #"Embeddings + Inferred_Commu": features_embeddings + features_commu_inferee,
    #"Structure + Inferred_Commu": features_structure + features_commu_inferee,
    #"Structure + Embeddings": features_structure + features_embeddings,
    "Full_features": features_structure + features_commu_inferee + features_embeddings
}

all_results = []

#for i in [0.00,1.00]:
for i in np.arange(1.00, -0.25, -0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv5_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    G_name_obj = f"artificial_graph_sbmv5_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    # 1. Chargement des données d'entraînement
    G_train, dataset_train, dataset_eval, _, _, _ = gp.load_all_data_for_graph(G_name)
    G = load_graphml_safe(f"graph_library/{G_name_obj}.graphml")

    if 'GroundTruth_JSON' in G.graph:
        print(f"[INIT] Extraction de la GT GroundTruth pour {G_name}...")
        gt_raw = json.loads(G.graph['GroundTruth_JSON'])
        
        GT = {'GT_sbm_matrix': np.array(gt_raw['GT_sbm_matrix']),
            'GT_pos': np.array(gt_raw['GT_pos']),
            'GT_sbm_id': np.array(gt_raw['GT_sbm_id']),
            'GT_degrees_sbm': np.array(gt_raw['GT_degrees_sbm']),
            'GT_degrees_spatial': np.array(gt_raw['GT_degrees_spatial'])
             }
        
        if 'P_matrix_JSON' in G.graph:
            print("P_matrix trouvée.")
            GT['GT_proba'] = np.array(json.loads(G.graph['P_matrix_JSON']))
    else:
        print("[WARNING] Aucune GroundTruth_JSON trouvée dans G.graph")
        GT = None

    pre_calculated_folds = gput._prepare_precalculated_folds(G_train, k=1, GroundTruth=GT)
    
    # 3. Lancement des tests
    for exp_name, feat_list in experiments.items():
        missing = set(feat_list) - set(dataset_train.columns)
        if missing:
            print(f" Exp {exp_name} : colonnes manquantes {missing}. Skip.")
            print(set(dataset_train.columns))
            continue

        print(f" Running: {exp_name} for SBM={i}")
        Params, summary_optim = k_fold_cross_validation_exp(G_train, pre_calculated_folds, feat_list, n_trials=30)
        
        stats_df, model, X_test, _, y_test, _ = gp.train_and_test_xgboost(dataset_train, features=feat_list,
                                                                          parameters = Params, plot=False)

        importances = model.feature_importances_
        feat_imp_series = pd.Series(importances, index=feat_list).sort_values(ascending=False)
        print(f"Top 5 Feature Importance pour {exp_name} :")
        print(feat_imp_series.head(5).to_string())

        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

        if exp_name == "GT_sbm":
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
output_path = os.path.join(output_dir, "comparaison_finale_structure_shuffled_v5_20260401.csv")
df_compare.to_csv(output_path, index=False)
print(f" Succès ! Fichier sauvegardé dans : {output_path}")
def plot_f1_comparison(df_results, metric="F1-Score", name ="OUBLI"):
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Création du lineplot
    # On utilise 'Experiment' pour différencier les lignes par couleur et style
    ax = sns.lineplot(
        data=df_results, 
        x="Ratio_SBM", 
        y= metric, 
        hue="Experiment", 
        style="Experiment", 
        markers=True, 
        dashes=False,
        linewidth=2.5,
        markersize=8
    )
    
    # Personnalisation des axes
    plt.title(f"Évolution du {metric} selon le ratio SBM vs POS", fontsize=15, pad=20)
    plt.xlabel("Ratio SBM (Structure de communauté forte → faible)", fontsize=12)
    plt.ylabel(f"{metric} (Link Prediction)", fontsize=12)
    
    # Inversion de l'axe X si tu veux montrer la difficulté croissante
    # plt.gca().invert_xaxis() 
    
    plt.legend(title="Configurations Features", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0.59, 0.95) # Pour bien voir l'échelle de 0 à 1
    
    plt.tight_layout()

    output_dir = os.path.join("outputs", "plots")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{name}_{metric}_per_SBM_ratio.png")
    
    # 3. La ligne magique
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"Graphique sauvegardé ici : {save_path}")
    
    plt.show()

df_compare = pd.read_csv("outputs/results/comparaison_finale_structure_shuffled_v5_20260401.csv")

plot_f1_comparison(df_compare, metric="AUC-ROC_eval", name = "artificial_graphs_shuffled_v5_20260401" )
def plot_f1_comparison(df_results, metric="F1-Score", name ="OUBLI"):
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Création du lineplot
    # On utilise 'Experiment' pour différencier les lignes par couleur et style
    ax = sns.lineplot(
        data=df_results, 
        x="Ratio_SBM", 
        y= metric, 
        hue="Experiment", 
        style="Experiment", 
        markers=True, 
        dashes=False,
        linewidth=2.5,
        markersize=8
    )
    
    # Personnalisation des axes
    plt.title(f"Évolution du {metric} selon le ratio SBM vs POS", fontsize=15, pad=20)
    plt.xlabel("Ratio SBM (Structure de communauté forte → faible)", fontsize=12)
    plt.ylabel(f"{metric} (Link Prediction)", fontsize=12)
    
    # Inversion de l'axe X si tu veux montrer la difficulté croissante
    # plt.gca().invert_xaxis() 
    
    plt.legend(title="Configurations Features", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0.89, 0.95) # Pour bien voir l'échelle de 0 à 1
    
    plt.tight_layout()

    output_dir = os.path.join("outputs", "plots")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{name}_{metric}_per_SBM_ratio.png")
    
    # 3. La ligne magique
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"Graphique sauvegardé ici : {save_path}")
    
    plt.show()

df_compare = pd.read_csv("outputs/results/comparaison_finale_structure_shuffled_v5_20260401.csv")

plot_f1_comparison(df_compare, metric="AUC-ROC_eval", name = "artificial_graphs_shuffled_v5_20260401" )
def plot_f1_comparison(df_results, metric="F1-Score", name ="OUBLI"):
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Création du lineplot
    # On utilise 'Experiment' pour différencier les lignes par couleur et style
    ax = sns.lineplot(
        data=df_results, 
        x="Ratio_SBM", 
        y= metric, 
        hue="Experiment", 
        style="Experiment", 
        markers=True, 
        dashes=False,
        linewidth=2.5,
        markersize=8
    )
    
    # Personnalisation des axes
    plt.title(f"Évolution du {metric} selon le ratio SBM vs POS", fontsize=15, pad=20)
    plt.xlabel("Ratio SBM (Structure de communauté forte → faible)", fontsize=12)
    plt.ylabel(f"{metric} (Link Prediction)", fontsize=12)
    
    # Inversion de l'axe X si tu veux montrer la difficulté croissante
    # plt.gca().invert_xaxis() 
    
    plt.legend(title="Configurations Features", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0.59, 0.95) # Pour bien voir l'échelle de 0 à 1
    
    plt.tight_layout()

    output_dir = os.path.join("outputs", "plots")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{name}_{metric}_per_SBM_ratio.png")
    
    # 3. La ligne magique
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"Graphique sauvegardé ici : {save_path}")
    
    plt.show()

df_compare = pd.read_csv("outputs/results/comparaison_finale_structure_shuffled_v5_20260401.csv")

plot_f1_comparison(df_compare, metric="AUC-ROC_eval", name = "artificial_graphs_shuffled_v5_20260401" )
all_summaries = []

for i in np.arange(0, 1.25, 0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv4_shuffled_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    _, _, _, xgboost_data, shap_explainer, _ = gp.load_all_data_for_graph(G_name)
    X_test = xgboost_data['X_test']

    print(f"Type de l'objet : {type(shap_explainer)}")
    print(f"Dimensions des valeurs : {shap_explainer.values.shape}")

    vals = shap_explainer.values
    
    if isinstance(vals, list):
        vals = vals[1]

    df_shap = pd.DataFrame(vals, columns=X_test.columns)
    mean_shap = df_shap.mean()
    mean_abs_shap = df_shap.abs().mean()
    
    summary = pd.DataFrame({
        'feature': X_test.columns,
        'mean_shap': mean_shap.values,
        'mean_abs_shap': mean_abs_shap.values,
        'sbm_param': i
    })
    all_summaries.append(summary)

df_shapvals_concat = pd.concat(all_summaries)
all_summaries = []

for i in np.arange(0, 1.25, 0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv4_shuffled_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}_GT"
    
    _, _, _, xgboost_data, shap_explainer, _ = gp.load_all_data_for_graph(G_name)
    X_test = xgboost_data['X_test']

    print(f"Type de l'objet : {type(shap_explainer)}")
    print(f"Dimensions des valeurs : {shap_explainer.values.shape}")

    vals = shap_explainer.values
    
    if isinstance(vals, list):
        vals = vals[1]

    df_shap = pd.DataFrame(vals, columns=X_test.columns)
    mean_shap = df_shap.mean()
    mean_abs_shap = df_shap.abs().mean()
    
    summary = pd.DataFrame({
        'feature': X_test.columns,
        'mean_shap': mean_shap.values,
        'mean_abs_shap': mean_abs_shap.values,
        'sbm_param': i
    })
    all_summaries.append(summary)

df_shapvals_concat = pd.concat(all_summaries)
all_summaries = []

for i in np.arange(0, 1.25, 0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv4_shuffled_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}_GT"
    
    _, _, _, xgboost_data, shap_explainer, _ = gp.load_all_data_for_graph(G_name)
    X_test = xgboost_data['X_test']

    print(f"Type de l'objet : {type(shap_explainer)}")
    print(f"Dimensions des valeurs : {shap_explainer.values.shape}")

    vals = shap_explainer.values
    
    if isinstance(vals, list):
        vals = vals[1]

    df_shap = pd.DataFrame(vals, columns=X_test.columns)
    mean_shap = df_shap.mean()
    mean_abs_shap = df_shap.abs().mean()
    
    summary = pd.DataFrame({
        'feature': X_test.columns,
        'mean_shap': mean_shap.values,
        'mean_abs_shap': mean_abs_shap.values,
        'sbm_param': i
    })
    all_summaries.append(summary)

df_shapvals_concat = pd.concat(all_summaries)
all_summaries = []

for i in np.arange(0, 1.25, 0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv4_shuffled_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}_GT.joblib"
    
    _, _, _, xgboost_data, shap_explainer, _ = gp.load_all_data_for_graph(G_name)
    X_test = xgboost_data['X_test']

    print(f"Type de l'objet : {type(shap_explainer)}")
    print(f"Dimensions des valeurs : {shap_explainer.values.shape}")

    vals = shap_explainer.values
    
    if isinstance(vals, list):
        vals = vals[1]

    df_shap = pd.DataFrame(vals, columns=X_test.columns)
    mean_shap = df_shap.mean()
    mean_abs_shap = df_shap.abs().mean()
    
    summary = pd.DataFrame({
        'feature': X_test.columns,
        'mean_shap': mean_shap.values,
        'mean_abs_shap': mean_abs_shap.values,
        'sbm_param': i
    })
    all_summaries.append(summary)

df_shapvals_concat = pd.concat(all_summaries)
all_summaries = []

for i in np.arange(0, 1.25, 0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv4_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}_GT"
    
    _, _, _, xgboost_data, shap_explainer, _ = gp.load_all_data_for_graph(G_name)
    X_test = xgboost_data['X_test']

    print(f"Type de l'objet : {type(shap_explainer)}")
    print(f"Dimensions des valeurs : {shap_explainer.values.shape}")

    vals = shap_explainer.values
    
    if isinstance(vals, list):
        vals = vals[1]

    df_shap = pd.DataFrame(vals, columns=X_test.columns)
    mean_shap = df_shap.mean()
    mean_abs_shap = df_shap.abs().mean()
    
    summary = pd.DataFrame({
        'feature': X_test.columns,
        'mean_shap': mean_shap.values,
        'mean_abs_shap': mean_abs_shap.values,
        'sbm_param': i
    })
    all_summaries.append(summary)

df_shapvals_concat = pd.concat(all_summaries)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_importance_evolution(df_results, metric="mean_shap", as_ratio=True):
    """
    Génère une heatmap montrant l'évolution de l'importance des features (SHAP).
    """
    heatmap_data = df_results.pivot(index='feature', columns='sbm_param', values=metric)
    
    if as_ratio:
        heatmap_data = heatmap_data.div(heatmap_data.sum(axis=0), axis=1)
        label_name = f"Ratio de contribution ({metric})"
        fmt_type = ".2%"
    else:
        label_name = f"Valeur brute ({metric})"
        fmt_type = ".3f"

    heatmap_data['avg_importance'] = heatmap_data.mean(axis=1)
    heatmap_data = heatmap_data.sort_values(by='avg_importance', ascending=False).drop(columns='avg_importance')

    plt.figure(figsize=(16, 9))
    sns.heatmap(heatmap_data, 
                annot=True,       
                fmt=fmt_type,        
                cmap="YlGnBu",    
                cbar_kws={'label': label_name})

    title_suffix = "(Ratios normalisés)" if as_ratio else "(Valeurs brutes)"
    plt.title(f"Importance relative des features (SHAP) selon la structure du graphe {title_suffix}")
    plt.xlabel("Paramètre du graphe (i)")
    plt.ylabel("Features")
    plt.tight_layout()

    filename = f"outputs/plots/heatmap_feat_importance_per_ratio_{metric}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé sous : {filename}")
    plt.show()

plot_feature_importance_evolution(df_shapvals_concat, metric="mean_abs_shap", as_ratio=False)
print(df_shapvals_concat.columns)
features_list = df_shapvals_concat['feature']
print(", ".join(map(str, features_list)))

"""
all_distribs = []
for i in np.arange(0, 1.05, 0.05):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbm_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    _, _,_, xgboost_data, _, _ = gp.load_all_data_for_graph(G_name)
    X_test = xgboost_data['X_test']
    y_test = xgboost_data['y_test']
    
    pos_sp = X_test[y_test == 1]['sp'].value_counts(normalize=True).to_dict()
    neg_sp = X_test[y_test == 0]['sp'].value_counts(normalize=True).to_dict()
    
    row = {'sbm_param': round(i, 2)}
    for dist in range(1, 7):
        row[f'pos_sp_{dist}'] = pos_sp.get(dist, 0)
        row[f'neg_sp_{dist}'] = neg_sp.get(dist, 0)
    
    row['pos_sp_7+'] = sum(v for k, v in pos_sp.items() if k >= 7 or k == 0)
    row['neg_sp_7+'] = sum(v for k, v in neg_sp.items() if k >= 7 or k == 0)
    
    all_distribs.append(row)

df_sp_detailed = pd.DataFrame(all_distribs)

cols = ['sbm_param'] + [f'{prefix}_sp_{d}' for d in list(range(1, 7)) + ['7+'] for prefix in ['pos', 'neg']]
df_sp_detailed = df_sp_detailed[cols]

pd.options.display.float_format = '{:.3f}'.format
print(df_sp_detailed.to_string(index=False))
"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_importance_evolution(df_results, metric="mean_shap", as_ratio=True):
    """
    Génère une heatmap montrant l'évolution de l'importance des features (SHAP).
    """
    heatmap_data = df_results.pivot(index='feature', columns='sbm_param', values=metric)
    
    if as_ratio:
        heatmap_data = heatmap_data.div(heatmap_data.sum(axis=0), axis=1)
        label_name = f"Ratio de contribution ({metric})"
        fmt_type = ".2%"
    else:
        label_name = f"Valeur brute ({metric})"
        fmt_type = ".3f"

    heatmap_data['avg_importance'] = heatmap_data.mean(axis=1)
    heatmap_data = heatmap_data.sort_values(by='avg_importance', ascending=False).drop(columns='avg_importance')

    plt.figure(figsize=(16, 9))
    sns.heatmap(heatmap_data, annot=True, fmt=fmt_type, cmap="YlGnBu", yticklabels=True, cbar_kws={'label': label_name})

    title_suffix = "(Ratios normalisés)" if as_ratio else "(Valeurs brutes)"
    plt.title(f"Importance relative des features (SHAP) selon la structure du graphe {title_suffix}")
    plt.xlabel("Paramètre du graphe (i)")
    plt.ylabel("Features")
    plt.tight_layout()

    filename = f"outputs/plots/heatmap_feat_importance_per_ratio_{metric}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé sous : {filename}")
    plt.show()

plot_feature_importance_evolution(df_shapvals_concat, metric="mean_abs_shap", as_ratio=False)
print(df_shapvals_concat.columns)
features_list = df_shapvals_concat['feature']
print(", ".join(map(str, features_list)))

"""
all_distribs = []
for i in np.arange(0, 1.05, 0.05):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbm_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    _, _,_, xgboost_data, _, _ = gp.load_all_data_for_graph(G_name)
    X_test = xgboost_data['X_test']
    y_test = xgboost_data['y_test']
    
    pos_sp = X_test[y_test == 1]['sp'].value_counts(normalize=True).to_dict()
    neg_sp = X_test[y_test == 0]['sp'].value_counts(normalize=True).to_dict()
    
    row = {'sbm_param': round(i, 2)}
    for dist in range(1, 7):
        row[f'pos_sp_{dist}'] = pos_sp.get(dist, 0)
        row[f'neg_sp_{dist}'] = neg_sp.get(dist, 0)
    
    row['pos_sp_7+'] = sum(v for k, v in pos_sp.items() if k >= 7 or k == 0)
    row['neg_sp_7+'] = sum(v for k, v in neg_sp.items() if k >= 7 or k == 0)
    
    all_distribs.append(row)

df_sp_detailed = pd.DataFrame(all_distribs)

cols = ['sbm_param'] + [f'{prefix}_sp_{d}' for d in list(range(1, 7)) + ['7+'] for prefix in ['pos', 'neg']]
df_sp_detailed = df_sp_detailed[cols]

pd.options.display.float_format = '{:.3f}'.format
print(df_sp_detailed.to_string(index=False))
"""
features_GT_proba = ['GT_proba']
features_GT_sbm = ['GT_sbm_density', 'GT_degrees_sbm_u', 'GT_degrees_sbm_v']
features_GT_pos = ['GT_pos_dist', 'GT_degrees_spatial_u','GT_degrees_spatial_v', 'GT_spatial_deg_product', 'GT_spatial_gravity_log']
features_structure = ['pr_u', 'pr_v','ppr_u', 'ppr_v', 'lcc_u', 'lcc_v', 'and_u', 'and_v', 'dc_u', 'dc_v', 'katz_u', 'katz_v',
                              'sp', 'jc', 'aa', 'cn','pa', 'ra']
features_commu_inferee = ['louvain_density', 'same_louvain', 'infomap_density', 'same_infomap', 'sbm_density', 'same_sbm',
                            'leiden_density', 'same_leiden', "surprise_density", "same_surprise", 
                            "significance_density", "same_significance"]
features_commu_surprise = ["surprise_density", "same_surprise"]
features_commu_infomap = ['infomap_density', 'same_infomap']
features_commu_sbm = ['sbm_density', 'same_sbm']
features_embeddings = ["deepwalk_cos", "deepwalk_rank", "deepwalk_had_mean",  "deepwalk_had_std",
                        "n2v_homophily_cos", "n2v_homophily_rank", "n2v_homophily_had_mean",  "n2v_homophily_had_std",
                      "crosswalk_cos", "crosswalk_rank", "crosswalk_had_mean",  "crosswalk_had_std"]

experiments = {
    "GT_absolue (proba)": features_GT_proba,
    "GT_sbm": features_GT_sbm,
    "GT_pos": features_GT_pos,
    #"GT": features_GT_sbm + features_GT_pos,
    "Structure_Only": features_structure,
    "Embeddings_Only": features_embeddings,
    "Inferred_Commu only": features_commu_inferee,
    #"Embeddings + Inferred_Commu": features_embeddings + features_commu_inferee,
    #"Structure + Inferred_Commu": features_structure + features_commu_inferee,
    #"Structure + Embeddings": features_structure + features_embeddings,
    "Full_features": features_structure + features_commu_inferee + features_embeddings
}

all_results = []

#for i in [0.00,1.00]:
for i in np.arange(1.00, -0.25, -0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv5_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    G_name_obj = f"artificial_graph_sbmv5_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    # 1. Chargement des données d'entraînement
    G_train, dataset_train, dataset_eval, _, _, _ = gp.load_all_data_for_graph(G_name)
    G = load_graphml_safe(f"graph_library/{G_name_obj}.graphml")

    if 'GroundTruth_JSON' in G.graph:
        print(f"[INIT] Extraction de la GT GroundTruth pour {G_name}...")
        gt_raw = json.loads(G.graph['GroundTruth_JSON'])
        
        GT = {'GT_sbm_matrix': np.array(gt_raw['GT_sbm_matrix']),
            'GT_pos': np.array(gt_raw['GT_pos']),
            'GT_sbm_id': np.array(gt_raw['GT_sbm_id']),
            'GT_degrees_sbm': np.array(gt_raw['GT_degrees_sbm']),
            'GT_degrees_spatial': np.array(gt_raw['GT_degrees_spatial'])
             }
        
        if 'P_matrix_JSON' in G.graph:
            print("P_matrix trouvée.")
            GT['GT_proba'] = np.array(json.loads(G.graph['P_matrix_JSON']))
    else:
        print("[WARNING] Aucune GroundTruth_JSON trouvée dans G.graph")
        GT = None

    pre_calculated_folds = gput._prepare_precalculated_folds(G_train, k=1, GroundTruth=GT)
    
    # 3. Lancement des tests
    for exp_name, feat_list in experiments.items():
        missing = set(feat_list) - set(dataset_train.columns)
        if missing:
            print(f" Exp {exp_name} : colonnes manquantes {missing}. Skip.")
            print(set(dataset_train.columns))
            continue

        print(f" Running: {exp_name} for SBM={i}")
        Params, summary_optim = k_fold_cross_validation_exp(G_train, pre_calculated_folds, feat_list, n_trials=30)
        
        stats_df, model, X_test, _, y_test, _ = gp.train_and_test_xgboost(dataset_train, features=feat_list,
                                                                          parameters = Params, plot=False)

        importances = model.feature_importances_
        feat_imp_series = pd.Series(importances, index=feat_list).sort_values(ascending=False)
        print(f"Top 5 Feature Importance pour {exp_name} :")
        print(feat_imp_series.head(5).to_string())

        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

        if exp_name == "GT_sbm":
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
output_path = os.path.join(output_dir, "comparaison_finale_structure_shuffled_v5_20260401.csv")
df_compare.to_csv(output_path, index=False)
print(f" Succès ! Fichier sauvegardé dans : {output_path}")
features_GT_proba = ['GT_proba']
features_GT_sbm = ['GT_sbm_density', 'GT_degrees_sbm_u', 'GT_degrees_sbm_v']
features_GT_pos = ['GT_pos_dist', 'GT_degrees_spatial_u','GT_degrees_spatial_v', 'GT_spatial_deg_product', 'GT_spatial_gravity_log']
features_structure = ['pr_u', 'pr_v','ppr_u', 'ppr_v', 'lcc_u', 'lcc_v', 'and_u', 'and_v', 'dc_u', 'dc_v', 'katz_u', 'katz_v',
                              'sp', 'jc', 'aa', 'cn','pa', 'ra']
features_commu_inferee = ['louvain_density', 'same_louvain', 'infomap_density', 'same_infomap', 'sbm_density', 'same_sbm',
                            'leiden_density', 'same_leiden', "surprise_density", "same_surprise", 
                            "significance_density", "same_significance"]
features_commu_surprise = ["surprise_density", "same_surprise"]
features_commu_infomap = ['infomap_density', 'same_infomap']
features_commu_sbm = ['sbm_density', 'same_sbm']
features_embeddings = ["deepwalk_cos", "deepwalk_rank", "deepwalk_had_mean",  "deepwalk_had_std",
                        "n2v_homophily_cos", "n2v_homophily_rank", "n2v_homophily_had_mean",  "n2v_homophily_had_std",
                      "crosswalk_cos", "crosswalk_rank", "crosswalk_had_mean",  "crosswalk_had_std"]

experiments = {
    "GT_absolue (proba)": features_GT_proba,
    "GT_sbm": features_GT_sbm,
    "GT_pos": features_GT_pos,
    #"GT": features_GT_sbm + features_GT_pos,
    "Structure_Only": features_structure,
    "Embeddings_Only": features_embeddings,
    "Inferred_Commu only": features_commu_inferee,
    #"Embeddings + Inferred_Commu": features_embeddings + features_commu_inferee,
    #"Structure + Inferred_Commu": features_structure + features_commu_inferee,
    #"Structure + Embeddings": features_structure + features_embeddings,
    "Full_features": features_structure + features_commu_inferee + features_embeddings
}

all_results = []

#for i in [0.00,1.00]:
for i in np.arange(1.00, -0.25, -0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv5_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    G_name_obj = f"artificial_graph_sbmv5_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    # 1. Chargement des données d'entraînement
    G_train, dataset_train, dataset_eval, _, _, _ = gp.load_all_data_for_graph(G_name)
    G = load_graphml_safe(f"graph_library/{G_name_obj}.graphml")

    if 'GroundTruth_JSON' in G.graph:
        print(f"[INIT] Extraction de la GT GroundTruth pour {G_name}...")
        gt_raw = json.loads(G.graph['GroundTruth_JSON'])
        
        GT = {'GT_sbm_matrix': np.array(gt_raw['GT_sbm_matrix']),
            'GT_pos': np.array(gt_raw['GT_pos']),
            'GT_sbm_id': np.array(gt_raw['GT_sbm_id']),
            'GT_degrees_sbm': np.array(gt_raw['GT_degrees_sbm']),
            'GT_degrees_spatial': np.array(gt_raw['GT_degrees_spatial'])
             }
        
        if 'P_matrix_JSON' in G.graph:
            print("P_matrix trouvée.")
            GT['GT_proba'] = np.array(json.loads(G.graph['P_matrix_JSON']))
    else:
        print("[WARNING] Aucune GroundTruth_JSON trouvée dans G.graph")
        GT = None

    pre_calculated_folds = gput._prepare_precalculated_folds(G_train, k=1, GroundTruth=GT)
    
    # 3. Lancement des tests
    for exp_name, feat_list in experiments.items():
        missing = set(feat_list) - set(dataset_train.columns)
        if missing:
            print(f" Exp {exp_name} : colonnes manquantes {missing}. Skip.")
            print(set(dataset_train.columns))
            continue

        print(f" Running: {exp_name} for SBM={i}")
        Params, summary_optim = k_fold_cross_validation_exp(G_train, pre_calculated_folds, feat_list, n_trials=30)
        
        stats_df, model, X_test, _, y_test, _ = gp.train_and_test_xgboost(dataset_train, features=feat_list,
                                                                          parameters = Params, plot=False)

        importances = model.feature_importances_
        feat_imp_series = pd.Series(importances, index=feat_list).sort_values(ascending=False)
        print(f"Top 5 Feature Importance pour {exp_name} :")
        print(feat_imp_series.head(5).to_string())

        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

        if exp_name == "GT_sbm":
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
output_path = os.path.join(output_dir, "comparaison_finale_structure_shuffled_v5_20260401.csv")
df_compare.to_csv(output_path, index=False)
print(f" Succès ! Fichier sauvegardé dans : {output_path}")
def plot_f1_comparison(df_results, metric="F1-Score", name ="OUBLI"):
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Création du lineplot
    # On utilise 'Experiment' pour différencier les lignes par couleur et style
    ax = sns.lineplot(
        data=df_results, 
        x="Ratio_SBM", 
        y= metric, 
        hue="Experiment", 
        style="Experiment", 
        markers=True, 
        dashes=False,
        linewidth=2.5,
        markersize=8
    )
    
    # Personnalisation des axes
    plt.title(f"Évolution du {metric} selon le ratio SBM vs POS", fontsize=15, pad=20)
    plt.xlabel("Ratio SBM (Structure de communauté forte → faible)", fontsize=12)
    plt.ylabel(f"{metric} (Link Prediction)", fontsize=12)
    
    # Inversion de l'axe X si tu veux montrer la difficulté croissante
    # plt.gca().invert_xaxis() 
    
    plt.legend(title="Configurations Features", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0.59, 0.95) # Pour bien voir l'échelle de 0 à 1
    
    plt.tight_layout()

    output_dir = os.path.join("outputs", "plots")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{name}_{metric}_per_SBM_ratio.png")
    
    # 3. La ligne magique
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"Graphique sauvegardé ici : {save_path}")
    
    plt.show()

df_compare = pd.read_csv("outputs/results/comparaison_finale_structure_shuffled_v5_20260401.csv")

plot_f1_comparison(df_compare, metric="AUC-ROC_eval", name = "artificial_graphs_shuffled_v5_20260401" )
features_GT_proba = ['GT_proba']
features_GT_sbm = ['GT_sbm_density', 'GT_degrees_sbm_u', 'GT_degrees_sbm_v']
features_GT_pos = ['GT_pos_dist', 'GT_degrees_spatial_u','GT_degrees_spatial_v', 'GT_spatial_deg_product', 'GT_spatial_gravity_log']
features_structure = ['pr_u', 'pr_v','ppr_u', 'ppr_v', 'lcc_u', 'lcc_v', 'and_u', 'and_v', 'dc_u', 'dc_v', 'katz_u', 'katz_v',
                              'sp', 'jc', 'aa', 'cn','pa', 'ra']
features_commu_inferee = ['louvain_density', 'same_louvain', 'infomap_density', 'same_infomap', 'sbm_density', 'same_sbm',
                            'leiden_density', 'same_leiden', "surprise_density", "same_surprise", 
                            "significance_density", "same_significance"]
features_commu_surprise = ["surprise_density", "same_surprise"]
features_commu_infomap = ['infomap_density', 'same_infomap']
features_commu_sbm = ['sbm_density', 'same_sbm']
features_embeddings = ["deepwalk_cos", "deepwalk_rank", "deepwalk_had_mean",  "deepwalk_had_std",
                        "n2v_homophily_cos", "n2v_homophily_rank", "n2v_homophily_had_mean",  "n2v_homophily_had_std",
                      "crosswalk_cos", "crosswalk_rank", "crosswalk_had_mean",  "crosswalk_had_std"]

experiments = {
    "GT_absolue (proba)": features_GT_proba,
    "GT_sbm": features_GT_sbm,
    "GT_pos": features_GT_pos,
    #"GT": features_GT_sbm + features_GT_pos,
    "Structure_Only": features_structure,
    "Embeddings_Only": features_embeddings,
    "Inferred_Commu only": features_commu_inferee,
    #"Embeddings + Inferred_Commu": features_embeddings + features_commu_inferee,
    #"Structure + Inferred_Commu": features_structure + features_commu_inferee,
    #"Structure + Embeddings": features_structure + features_embeddings,
    "Full_features": features_structure + features_commu_inferee + features_embeddings
}

all_results = []

#for i in [0.00,1.00]:
for i in np.arange(1.00, -0.25, -0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv5_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    G_name_obj = f"artificial_graph_sbmv5_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    # 1. Chargement des données d'entraînement
    G_train, dataset_train, dataset_eval, _, _, _ = gp.load_all_data_for_graph(G_name)
    G = load_graphml_safe(f"graph_library/{G_name_obj}.graphml")

    if 'GroundTruth_JSON' in G.graph:
        print(f"[INIT] Extraction de la GT GroundTruth pour {G_name}...")
        gt_raw = json.loads(G.graph['GroundTruth_JSON'])
        
        GT = {'GT_sbm_matrix': np.array(gt_raw['GT_sbm_matrix']),
            'GT_pos': np.array(gt_raw['GT_pos']),
            'GT_sbm_id': np.array(gt_raw['GT_sbm_id']),
            'GT_degrees_sbm': np.array(gt_raw['GT_degrees_sbm']),
            'GT_degrees_spatial': np.array(gt_raw['GT_degrees_spatial'])
             }
        
        if 'P_matrix_JSON' in G.graph:
            print("P_matrix trouvée.")
            GT['GT_proba'] = np.array(json.loads(G.graph['P_matrix_JSON']))
    else:
        print("[WARNING] Aucune GroundTruth_JSON trouvée dans G.graph")
        GT = None

    pre_calculated_folds = gput._prepare_precalculated_folds(G_train, k=1, GroundTruth=GT)
    
    # 3. Lancement des tests
    for exp_name, feat_list in experiments.items():
        missing = set(feat_list) - set(dataset_train.columns)
        if missing:
            print(f" Exp {exp_name} : colonnes manquantes {missing}. Skip.")
            print(set(dataset_train.columns))
            continue

        print(f" Running: {exp_name} for SBM={i}")
        Params, summary_optim = k_fold_cross_validation_exp(G_train, pre_calculated_folds, feat_list, n_trials=30)
        
        stats_df, model, X_test, _, y_test, _ = gp.train_and_test_xgboost(dataset_train, features=feat_list,
                                                                          parameters = Params, plot=False)

        importances = model.feature_importances_
        feat_imp_series = pd.Series(importances, index=feat_list).sort_values(ascending=False)
        print(f"Top 5 Feature Importance pour {exp_name} :")
        print(feat_imp_series.head(5).to_string())

        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

        if exp_name == "GT_sbm":
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
output_path = os.path.join(output_dir, "comparaison_finale_structure_shuffled_v5_20260401.csv")
df_compare.to_csv(output_path, index=False)
print(f" Succès ! Fichier sauvegardé dans : {output_path}")
def plot_f1_comparison(df_results, metric="F1-Score", name ="OUBLI"):
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Création du lineplot
    # On utilise 'Experiment' pour différencier les lignes par couleur et style
    ax = sns.lineplot(
        data=df_results, 
        x="Ratio_SBM", 
        y= metric, 
        hue="Experiment", 
        style="Experiment", 
        markers=True, 
        dashes=False,
        linewidth=2.5,
        markersize=8
    )
    
    # Personnalisation des axes
    plt.title(f"Évolution du {metric} selon le ratio SBM vs POS", fontsize=15, pad=20)
    plt.xlabel("Ratio SBM (Structure de communauté forte → faible)", fontsize=12)
    plt.ylabel(f"{metric} (Link Prediction)", fontsize=12)
    
    # Inversion de l'axe X si tu veux montrer la difficulté croissante
    # plt.gca().invert_xaxis() 
    
    plt.legend(title="Configurations Features", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0.59, 0.95) # Pour bien voir l'échelle de 0 à 1
    
    plt.tight_layout()

    output_dir = os.path.join("outputs", "plots")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{name}_{metric}_per_SBM_ratio.png")
    
    # 3. La ligne magique
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"Graphique sauvegardé ici : {save_path}")
    
    plt.show()

df_compare = pd.read_csv("outputs/results/comparaison_finale_structure_shuffled_v5_20260401.csv")

plot_f1_comparison(df_compare, metric="AUC-ROC_eval", name = "artificial_graphs_shuffled_v5_20260401" )
def plot_f1_comparison(df_results, metric="F1-Score", name ="OUBLI"):
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Création du lineplot
    # On utilise 'Experiment' pour différencier les lignes par couleur et style
    ax = sns.lineplot(
        data=df_results, 
        x="Ratio_SBM", 
        y= metric, 
        hue="Experiment", 
        style="Experiment", 
        markers=True, 
        dashes=False,
        linewidth=2.5,
        markersize=8
    )
    
    # Personnalisation des axes
    plt.title(f"Évolution du {metric} selon le ratio SBM vs POS", fontsize=15, pad=20)
    plt.xlabel("Ratio SBM (Structure de communauté forte → faible)", fontsize=12)
    plt.ylabel(f"{metric} (Link Prediction)", fontsize=12)
    
    # Inversion de l'axe X si tu veux montrer la difficulté croissante
    # plt.gca().invert_xaxis() 
    
    plt.legend(title="Configurations Features", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0.59, 0.95) # Pour bien voir l'échelle de 0 à 1
    
    plt.tight_layout()

    output_dir = os.path.join("outputs", "plots")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{name}_{metric}_per_SBM_ratio.png")
    
    # 3. La ligne magique
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"Graphique sauvegardé ici : {save_path}")
    
    plt.show()

df_compare = pd.read_csv("outputs/results/comparaison_finale_structure_shuffled_v5_20260401.csv")

plot_f1_comparison(df_compare, metric="AUC-ROC_eval", name = "artificial_graphs_shuffled_v5_20260401" )
%load_ext autoreload
%autoreload 2

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
features_GT_proba = ['GT_proba']
features_GT_sbm = ['GT_sbm_density', 'GT_degrees_sbm_u', 'GT_degrees_sbm_v']
features_GT_pos = ['GT_pos_dist', 'GT_degrees_spatial_u','GT_degrees_spatial_v', 'GT_spatial_deg_product', 'GT_spatial_gravity_log']
features_structure = ['pr_u', 'pr_v','ppr_u', 'ppr_v', 'lcc_u', 'lcc_v', 'and_u', 'and_v', 'dc_u', 'dc_v', 'katz_u', 'katz_v',
                              'sp', 'jc', 'aa', 'cn','pa', 'ra']
features_commu_inferee = ['louvain_density', 'same_louvain', 'infomap_density', 'same_infomap', 'sbm_density', 'same_sbm',
                            'leiden_density', 'same_leiden', "surprise_density", "same_surprise", 
                            "significance_density", "same_significance"]
features_commu_surprise = ["surprise_density", "same_surprise"]
features_commu_infomap = ['infomap_density', 'same_infomap']
features_commu_sbm = ['sbm_density', 'same_sbm']
features_embeddings = ["deepwalk_cos", "deepwalk_rank", "deepwalk_had_mean",  "deepwalk_had_std",
                        "n2v_homophily_cos", "n2v_homophily_rank", "n2v_homophily_had_mean",  "n2v_homophily_had_std",
                      "crosswalk_cos", "crosswalk_rank", "crosswalk_had_mean",  "crosswalk_had_std"]

experiments = {
    "GT_absolue (proba)": features_GT_proba,
    "GT_sbm": features_GT_sbm,
    "GT_pos": features_GT_pos,
    #"GT": features_GT_sbm + features_GT_pos,
    "Structure_Only": features_structure,
    "Embeddings_Only": features_embeddings,
    "Inferred_Commu only": features_commu_inferee,
    #"Embeddings + Inferred_Commu": features_embeddings + features_commu_inferee,
    #"Structure + Inferred_Commu": features_structure + features_commu_inferee,
    #"Structure + Embeddings": features_structure + features_embeddings,
    "Full_features": features_structure + features_commu_inferee + features_embeddings
}

all_results = []

#for i in [0.00,1.00]:
for i in np.arange(1.00, -0.25, -0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv5_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    G_name_obj = f"artificial_graph_sbmv5_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    # 1. Chargement des données d'entraînement
    G_train, dataset_train, dataset_eval, _, _, _ = gp.load_all_data_for_graph(G_name)
    G = load_graphml_safe(f"graph_library/{G_name_obj}.graphml")

    if 'GroundTruth_JSON' in G.graph:
        print(f"[INIT] Extraction de la GT GroundTruth pour {G_name}...")
        gt_raw = json.loads(G.graph['GroundTruth_JSON'])
        
        GT = {'GT_sbm_matrix': np.array(gt_raw['GT_sbm_matrix']),
            'GT_pos': np.array(gt_raw['GT_pos']),
            'GT_sbm_id': np.array(gt_raw['GT_sbm_id']),
            'GT_degrees_sbm': np.array(gt_raw['GT_degrees_sbm']),
            'GT_degrees_spatial': np.array(gt_raw['GT_degrees_spatial'])
             }
        
        if 'P_matrix_JSON' in G.graph:
            print("P_matrix trouvée.")
            GT['GT_proba'] = np.array(json.loads(G.graph['P_matrix_JSON']))
    else:
        print("[WARNING] Aucune GroundTruth_JSON trouvée dans G.graph")
        GT = None

    pre_calculated_folds = gput._prepare_precalculated_folds(G_train, k=1, GroundTruth=GT)
    
    # 3. Lancement des tests
    for exp_name, feat_list in experiments.items():
        missing = set(feat_list) - set(dataset_train.columns)
        if missing:
            print(f" Exp {exp_name} : colonnes manquantes {missing}. Skip.")
            print(set(dataset_train.columns))
            continue

        print(f" Running: {exp_name} for SBM={i}")
        Params, summary_optim = k_fold_cross_validation_exp(G_train, pre_calculated_folds, feat_list, n_trials=30)
        
        stats_df, model, X_test, _, y_test, _ = gp.train_and_test_xgboost(dataset_train, features=feat_list,
                                                                          parameters = Params, plot=False)

        importances = model.feature_importances_
        feat_imp_series = pd.Series(importances, index=feat_list).sort_values(ascending=False)
        print(f"Top 5 Feature Importance pour {exp_name} :")
        print(feat_imp_series.head(5).to_string())

        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

        if exp_name == "GT_sbm":
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
output_path = os.path.join(output_dir, "comparaison_finale_structure_shuffled_v5_20260401.csv")
df_compare.to_csv(output_path, index=False)
print(f" Succès ! Fichier sauvegardé dans : {output_path}")
features_GT_proba = ['GT_proba']
features_GT_sbm = ['GT_sbm_density', 'GT_degrees_sbm_u', 'GT_degrees_sbm_v']
features_GT_pos = ['GT_pos_dist', 'GT_degrees_spatial_u','GT_degrees_spatial_v', 'GT_spatial_deg_product', 
                   #'GT_spatial_gravity_log'
                  ]
features_structure = ['pr_u', 'pr_v','ppr_u', 'ppr_v', 'lcc_u', 'lcc_v', 'and_u', 'and_v', 'dc_u', 'dc_v', 'katz_u', 'katz_v',
                              'sp', 'jc', 'aa', 'cn','pa', 'ra']
features_commu_inferee = ['louvain_density', 'same_louvain', 'infomap_density', 'same_infomap', 'sbm_density', 'same_sbm',
                            'leiden_density', 'same_leiden', "surprise_density", "same_surprise", 
                            "significance_density", "same_significance"]
features_commu_surprise = ["surprise_density", "same_surprise"]
features_commu_infomap = ['infomap_density', 'same_infomap']
features_commu_sbm = ['sbm_density', 'same_sbm']
features_embeddings = ["deepwalk_cos", "deepwalk_rank", "deepwalk_had_mean",  "deepwalk_had_std",
                        "n2v_homophily_cos", "n2v_homophily_rank", "n2v_homophily_had_mean",  "n2v_homophily_had_std",
                      "crosswalk_cos", "crosswalk_rank", "crosswalk_had_mean",  "crosswalk_had_std"]

experiments = {
    "GT_absolue (proba)": features_GT_proba,
    "GT_sbm": features_GT_sbm,
    "GT_pos": features_GT_pos,
    #"GT": features_GT_sbm + features_GT_pos,
    "Structure_Only": features_structure,
    "Embeddings_Only": features_embeddings,
    "Inferred_Commu only": features_commu_inferee,
    #"Embeddings + Inferred_Commu": features_embeddings + features_commu_inferee,
    #"Structure + Inferred_Commu": features_structure + features_commu_inferee,
    #"Structure + Embeddings": features_structure + features_embeddings,
    "Full_features": features_structure + features_commu_inferee + features_embeddings
}

all_results = []

#for i in [0.00,1.00]:
for i in np.arange(1.00, -0.25, -0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv5_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    G_name_obj = f"artificial_graph_sbmv5_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    # 1. Chargement des données d'entraînement
    G_train, dataset_train, dataset_eval, _, _, _ = gp.load_all_data_for_graph(G_name)
    G = load_graphml_safe(f"graph_library/{G_name_obj}.graphml")

    if 'GroundTruth_JSON' in G.graph:
        print(f"[INIT] Extraction de la GT GroundTruth pour {G_name}...")
        gt_raw = json.loads(G.graph['GroundTruth_JSON'])
        
        GT = {'GT_sbm_matrix': np.array(gt_raw['GT_sbm_matrix']),
            'GT_pos': np.array(gt_raw['GT_pos']),
            'GT_sbm_id': np.array(gt_raw['GT_sbm_id']),
            'GT_degrees_sbm': np.array(gt_raw['GT_degrees_sbm']),
            'GT_degrees_spatial': np.array(gt_raw['GT_degrees_spatial'])
             }
        
        if 'P_matrix_JSON' in G.graph:
            print("P_matrix trouvée.")
            GT['GT_proba'] = np.array(json.loads(G.graph['P_matrix_JSON']))
    else:
        print("[WARNING] Aucune GroundTruth_JSON trouvée dans G.graph")
        GT = None

    pre_calculated_folds = gput._prepare_precalculated_folds(G_train, k=1, GroundTruth=GT)
    
    # 3. Lancement des tests
    for exp_name, feat_list in experiments.items():
        missing = set(feat_list) - set(dataset_train.columns)
        if missing:
            print(f" Exp {exp_name} : colonnes manquantes {missing}. Skip.")
            print(set(dataset_train.columns))
            continue

        print(f" Running: {exp_name} for SBM={i}")
        Params, summary_optim = k_fold_cross_validation_exp(G_train, pre_calculated_folds, feat_list, n_trials=30)
        
        stats_df, model, X_test, _, y_test, _ = gp.train_and_test_xgboost(dataset_train, features=feat_list,
                                                                          parameters = Params, plot=False)

        importances = model.feature_importances_
        feat_imp_series = pd.Series(importances, index=feat_list).sort_values(ascending=False)
        print(f"Top 5 Feature Importance pour {exp_name} :")
        print(feat_imp_series.head(5).to_string())

        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

        if exp_name == "GT_sbm":
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
output_path = os.path.join(output_dir, "comparaison_finale_structure_shuffled_v5_20260401.csv")
df_compare.to_csv(output_path, index=False)
print(f" Succès ! Fichier sauvegardé dans : {output_path}")
def plot_f1_comparison(df_results, metric="F1-Score", name ="OUBLI"):
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Création du lineplot
    # On utilise 'Experiment' pour différencier les lignes par couleur et style
    ax = sns.lineplot(
        data=df_results, 
        x="Ratio_SBM", 
        y= metric, 
        hue="Experiment", 
        style="Experiment", 
        markers=True, 
        dashes=False,
        linewidth=2.5,
        markersize=8
    )
    
    # Personnalisation des axes
    plt.title(f"Évolution du {metric} selon le ratio SBM vs POS", fontsize=15, pad=20)
    plt.xlabel("Ratio SBM (Structure de communauté forte → faible)", fontsize=12)
    plt.ylabel(f"{metric} (Link Prediction)", fontsize=12)
    
    # Inversion de l'axe X si tu veux montrer la difficulté croissante
    # plt.gca().invert_xaxis() 
    
    plt.legend(title="Configurations Features", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0.59, 0.95) # Pour bien voir l'échelle de 0 à 1
    
    plt.tight_layout()

    output_dir = os.path.join("outputs", "plots")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{name}_{metric}_per_SBM_ratio.png")
    
    # 3. La ligne magique
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"Graphique sauvegardé ici : {save_path}")
    
    plt.show()

df_compare = pd.read_csv("outputs/results/comparaison_finale_structure_shuffled_v5_20260401.csv")

plot_f1_comparison(df_compare, metric="AUC-ROC_eval", name = "artificial_graphs_shuffled_v5_20260401" )
all_summaries = []

for i in np.arange(0, 1.25, 0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv4_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}_GT"
    
    _, _, _, xgboost_data, shap_explainer, _ = gp.load_all_data_for_graph(G_name)
    X_test = xgboost_data['X_test']

    print(f"Type de l'objet : {type(shap_explainer)}")
    print(f"Dimensions des valeurs : {shap_explainer.values.shape}")

    vals = shap_explainer.values
    
    if isinstance(vals, list):
        vals = vals[1]

    df_shap = pd.DataFrame(vals, columns=X_test.columns)
    mean_shap = df_shap.mean()
    mean_abs_shap = df_shap.abs().mean()
    
    summary = pd.DataFrame({
        'feature': X_test.columns,
        'mean_shap': mean_shap.values,
        'mean_abs_shap': mean_abs_shap.values,
        'sbm_param': i
    })
    all_summaries.append(summary)

df_shapvals_concat = pd.concat(all_summaries)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_importance_evolution(df_results, metric="mean_shap", as_ratio=True):
    """
    Génère une heatmap montrant l'évolution de l'importance des features (SHAP).
    """
    heatmap_data = df_results.pivot(index='feature', columns='sbm_param', values=metric)
    
    if as_ratio:
        heatmap_data = heatmap_data.div(heatmap_data.sum(axis=0), axis=1)
        label_name = f"Ratio de contribution ({metric})"
        fmt_type = ".2%"
    else:
        label_name = f"Valeur brute ({metric})"
        fmt_type = ".3f"

    heatmap_data['avg_importance'] = heatmap_data.mean(axis=1)
    heatmap_data = heatmap_data.sort_values(by='avg_importance', ascending=False).drop(columns='avg_importance')

    plt.figure(figsize=(16, 9))
    sns.heatmap(heatmap_data, annot=True, fmt=fmt_type, cmap="YlGnBu", yticklabels=True, cbar_kws={'label': label_name})

    title_suffix = "(Ratios normalisés)" if as_ratio else "(Valeurs brutes)"
    plt.title(f"Importance relative des features (SHAP) selon la structure du graphe {title_suffix}")
    plt.xlabel("Paramètre du graphe (i)")
    plt.ylabel("Features")
    plt.tight_layout()

    filename = f"outputs/plots/heatmap_feat_importance_per_ratio_{metric}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé sous : {filename}")
    plt.show()

plot_feature_importance_evolution(df_shapvals_concat, metric="mean_abs_shap", as_ratio=False)
print(df_shapvals_concat.columns)
features_list = df_shapvals_concat['feature']
print(", ".join(map(str, features_list)))

"""
all_distribs = []
for i in np.arange(0, 1.05, 0.05):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbm_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    _, _,_, xgboost_data, _, _ = gp.load_all_data_for_graph(G_name)
    X_test = xgboost_data['X_test']
    y_test = xgboost_data['y_test']
    
    pos_sp = X_test[y_test == 1]['sp'].value_counts(normalize=True).to_dict()
    neg_sp = X_test[y_test == 0]['sp'].value_counts(normalize=True).to_dict()
    
    row = {'sbm_param': round(i, 2)}
    for dist in range(1, 7):
        row[f'pos_sp_{dist}'] = pos_sp.get(dist, 0)
        row[f'neg_sp_{dist}'] = neg_sp.get(dist, 0)
    
    row['pos_sp_7+'] = sum(v for k, v in pos_sp.items() if k >= 7 or k == 0)
    row['neg_sp_7+'] = sum(v for k, v in neg_sp.items() if k >= 7 or k == 0)
    
    all_distribs.append(row)

df_sp_detailed = pd.DataFrame(all_distribs)

cols = ['sbm_param'] + [f'{prefix}_sp_{d}' for d in list(range(1, 7)) + ['7+'] for prefix in ['pos', 'neg']]
df_sp_detailed = df_sp_detailed[cols]

pd.options.display.float_format = '{:.3f}'.format
print(df_sp_detailed.to_string(index=False))
"""
all_summaries = []

for i in np.arange(0, 1.25, 0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv5_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}_GT"
    
    _, _, _, xgboost_data, shap_explainer, _ = gp.load_all_data_for_graph(G_name)
    X_test = xgboost_data['X_test']

    print(f"Type de l'objet : {type(shap_explainer)}")
    print(f"Dimensions des valeurs : {shap_explainer.values.shape}")

    vals = shap_explainer.values
    
    if isinstance(vals, list):
        vals = vals[1]

    df_shap = pd.DataFrame(vals, columns=X_test.columns)
    mean_shap = df_shap.mean()
    mean_abs_shap = df_shap.abs().mean()
    
    summary = pd.DataFrame({
        'feature': X_test.columns,
        'mean_shap': mean_shap.values,
        'mean_abs_shap': mean_abs_shap.values,
        'sbm_param': i
    })
    all_summaries.append(summary)

df_shapvals_concat = pd.concat(all_summaries)
all_summaries = []

for i in np.arange(0, 1.25, 0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv5_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    _, _, _, xgboost_data, shap_explainer, _ = gp.load_all_data_for_graph(G_name)
    X_test = xgboost_data['X_test']

    print(f"Type de l'objet : {type(shap_explainer)}")
    print(f"Dimensions des valeurs : {shap_explainer.values.shape}")

    vals = shap_explainer.values
    
    if isinstance(vals, list):
        vals = vals[1]

    df_shap = pd.DataFrame(vals, columns=X_test.columns)
    mean_shap = df_shap.mean()
    mean_abs_shap = df_shap.abs().mean()
    
    summary = pd.DataFrame({
        'feature': X_test.columns,
        'mean_shap': mean_shap.values,
        'mean_abs_shap': mean_abs_shap.values,
        'sbm_param': i
    })
    all_summaries.append(summary)

df_shapvals_concat = pd.concat(all_summaries)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_importance_evolution(df_results, metric="mean_shap", as_ratio=True):
    """
    Génère une heatmap montrant l'évolution de l'importance des features (SHAP).
    """
    heatmap_data = df_results.pivot(index='feature', columns='sbm_param', values=metric)
    
    if as_ratio:
        heatmap_data = heatmap_data.div(heatmap_data.sum(axis=0), axis=1)
        label_name = f"Ratio de contribution ({metric})"
        fmt_type = ".2%"
    else:
        label_name = f"Valeur brute ({metric})"
        fmt_type = ".3f"

    heatmap_data['avg_importance'] = heatmap_data.mean(axis=1)
    heatmap_data = heatmap_data.sort_values(by='avg_importance', ascending=False).drop(columns='avg_importance')

    plt.figure(figsize=(16, 9))
    sns.heatmap(heatmap_data, annot=True, fmt=fmt_type, cmap="YlGnBu", yticklabels=True, cbar_kws={'label': label_name})

    title_suffix = "(Ratios normalisés)" if as_ratio else "(Valeurs brutes)"
    plt.title(f"Importance relative des features (SHAP) selon la structure du graphe {title_suffix}")
    plt.xlabel("Paramètre du graphe (i)")
    plt.ylabel("Features")
    plt.tight_layout()

    filename = f"outputs/plots/heatmap_feat_importance_per_ratio_{metric}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé sous : {filename}")
    plt.show()

plot_feature_importance_evolution(df_shapvals_concat, metric="mean_abs_shap", as_ratio=False)
print(df_shapvals_concat.columns)
features_list = df_shapvals_concat['feature']
print(", ".join(map(str, features_list)))

"""
all_distribs = []
for i in np.arange(0, 1.05, 0.05):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbm_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    _, _,_, xgboost_data, _, _ = gp.load_all_data_for_graph(G_name)
    X_test = xgboost_data['X_test']
    y_test = xgboost_data['y_test']
    
    pos_sp = X_test[y_test == 1]['sp'].value_counts(normalize=True).to_dict()
    neg_sp = X_test[y_test == 0]['sp'].value_counts(normalize=True).to_dict()
    
    row = {'sbm_param': round(i, 2)}
    for dist in range(1, 7):
        row[f'pos_sp_{dist}'] = pos_sp.get(dist, 0)
        row[f'neg_sp_{dist}'] = neg_sp.get(dist, 0)
    
    row['pos_sp_7+'] = sum(v for k, v in pos_sp.items() if k >= 7 or k == 0)
    row['neg_sp_7+'] = sum(v for k, v in neg_sp.items() if k >= 7 or k == 0)
    
    all_distribs.append(row)

df_sp_detailed = pd.DataFrame(all_distribs)

cols = ['sbm_param'] + [f'{prefix}_sp_{d}' for d in list(range(1, 7)) + ['7+'] for prefix in ['pos', 'neg']]
df_sp_detailed = df_sp_detailed[cols]

pd.options.display.float_format = '{:.3f}'.format
print(df_sp_detailed.to_string(index=False))
"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_importance_evolution(df_results, metric="mean_shap", as_ratio=True):
    """
    Génère une heatmap montrant l'évolution de l'importance des features (SHAP).
    """
    heatmap_data = df_results.pivot(index='feature', columns='sbm_param', values=metric)
    
    if as_ratio:
        heatmap_data = heatmap_data.div(heatmap_data.sum(axis=0), axis=1)
        label_name = f"Ratio de contribution ({metric})"
        fmt_type = ".2%"
    else:
        label_name = f"Valeur brute ({metric})"
        fmt_type = ".3f"

    heatmap_data['avg_importance'] = heatmap_data.mean(axis=1)
    heatmap_data = heatmap_data.sort_values(by='avg_importance', ascending=False).drop(columns='avg_importance')

    plt.figure(figsize=(16, 9))
    sns.heatmap(heatmap_data, annot=True, fmt=fmt_type, cmap="YlGnBu", yticklabels=True, cbar_kws={'label': label_name})

    title_suffix = "(Ratios normalisés)" if as_ratio else "(Valeurs brutes)"
    plt.title(f"Importance relative des features (SHAP) selon la structure du graphe {title_suffix}")
    plt.xlabel("Paramètre du graphe (i)")
    plt.ylabel("Features")
    plt.tight_layout()

    filename = f"outputs/plots/heatmap_feat_importance_per_ratio_v5_{metric}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé sous : {filename}")
    plt.show()

plot_feature_importance_evolution(df_shapvals_concat, metric="mean_abs_shap", as_ratio=False)
print(df_shapvals_concat.columns)
features_list = df_shapvals_concat['feature']
print(", ".join(map(str, features_list)))

"""
all_distribs = []
for i in np.arange(0, 1.05, 0.05):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbm_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    _, _,_, xgboost_data, _, _ = gp.load_all_data_for_graph(G_name)
    X_test = xgboost_data['X_test']
    y_test = xgboost_data['y_test']
    
    pos_sp = X_test[y_test == 1]['sp'].value_counts(normalize=True).to_dict()
    neg_sp = X_test[y_test == 0]['sp'].value_counts(normalize=True).to_dict()
    
    row = {'sbm_param': round(i, 2)}
    for dist in range(1, 7):
        row[f'pos_sp_{dist}'] = pos_sp.get(dist, 0)
        row[f'neg_sp_{dist}'] = neg_sp.get(dist, 0)
    
    row['pos_sp_7+'] = sum(v for k, v in pos_sp.items() if k >= 7 or k == 0)
    row['neg_sp_7+'] = sum(v for k, v in neg_sp.items() if k >= 7 or k == 0)
    
    all_distribs.append(row)

df_sp_detailed = pd.DataFrame(all_distribs)

cols = ['sbm_param'] + [f'{prefix}_sp_{d}' for d in list(range(1, 7)) + ['7+'] for prefix in ['pos', 'neg']]
df_sp_detailed = df_sp_detailed[cols]

pd.options.display.float_format = '{:.3f}'.format
print(df_sp_detailed.to_string(index=False))
"""
all_summaries = []

for i in np.arange(0, 1.25, 0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv4_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    _, _, _, xgboost_data, shap_explainer, _ = gp.load_all_data_for_graph(G_name)
    X_test = xgboost_data['X_test']

    print(f"Type de l'objet : {type(shap_explainer)}")
    print(f"Dimensions des valeurs : {shap_explainer.values.shape}")

    vals = shap_explainer.values
    
    if isinstance(vals, list):
        vals = vals[1]

    df_shap = pd.DataFrame(vals, columns=X_test.columns)
    mean_shap = df_shap.mean()
    mean_abs_shap = df_shap.abs().mean()
    
    summary = pd.DataFrame({
        'feature': X_test.columns,
        'mean_shap': mean_shap.values,
        'mean_abs_shap': mean_abs_shap.values,
        'sbm_param': i
    })
    all_summaries.append(summary)

df_shapvals_concat = pd.concat(all_summaries)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_importance_evolution(df_results, metric="mean_shap", as_ratio=True):
    """
    Génère une heatmap montrant l'évolution de l'importance des features (SHAP).
    """
    heatmap_data = df_results.pivot(index='feature', columns='sbm_param', values=metric)
    
    if as_ratio:
        heatmap_data = heatmap_data.div(heatmap_data.sum(axis=0), axis=1)
        label_name = f"Ratio de contribution ({metric})"
        fmt_type = ".2%"
    else:
        label_name = f"Valeur brute ({metric})"
        fmt_type = ".3f"

    heatmap_data['avg_importance'] = heatmap_data.mean(axis=1)
    heatmap_data = heatmap_data.sort_values(by='avg_importance', ascending=False).drop(columns='avg_importance')

    plt.figure(figsize=(16, 9))
    sns.heatmap(heatmap_data, annot=True, fmt=fmt_type, cmap="YlGnBu", yticklabels=True, cbar_kws={'label': label_name})

    title_suffix = "(Ratios normalisés)" if as_ratio else "(Valeurs brutes)"
    plt.title(f"Importance relative des features (SHAP) selon la structure du graphe {title_suffix}")
    plt.xlabel("Paramètre du graphe (i)")
    plt.ylabel("Features")
    plt.tight_layout()

    filename = f"outputs/plots/heatmap_feat_importance_per_ratio_v4_{metric}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé sous : {filename}")
    plt.show()

plot_feature_importance_evolution(df_shapvals_concat, metric="mean_abs_shap", as_ratio=False)
print(df_shapvals_concat.columns)
features_list = df_shapvals_concat['feature']
print(", ".join(map(str, features_list)))

"""
all_distribs = []
for i in np.arange(0, 1.05, 0.05):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbm_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    _, _,_, xgboost_data, _, _ = gp.load_all_data_for_graph(G_name)
    X_test = xgboost_data['X_test']
    y_test = xgboost_data['y_test']
    
    pos_sp = X_test[y_test == 1]['sp'].value_counts(normalize=True).to_dict()
    neg_sp = X_test[y_test == 0]['sp'].value_counts(normalize=True).to_dict()
    
    row = {'sbm_param': round(i, 2)}
    for dist in range(1, 7):
        row[f'pos_sp_{dist}'] = pos_sp.get(dist, 0)
        row[f'neg_sp_{dist}'] = neg_sp.get(dist, 0)
    
    row['pos_sp_7+'] = sum(v for k, v in pos_sp.items() if k >= 7 or k == 0)
    row['neg_sp_7+'] = sum(v for k, v in neg_sp.items() if k >= 7 or k == 0)
    
    all_distribs.append(row)

df_sp_detailed = pd.DataFrame(all_distribs)

cols = ['sbm_param'] + [f'{prefix}_sp_{d}' for d in list(range(1, 7)) + ['7+'] for prefix in ['pos', 'neg']]
df_sp_detailed = df_sp_detailed[cols]

pd.options.display.float_format = '{:.3f}'.format
print(df_sp_detailed.to_string(index=False))
"""
features_GT = ['GT_sbm_density', 'GT_degrees_sbm_u', 'GT_degrees_sbm_v',
               'GT_pos_dist', 'GT_degrees_spatial_u','GT_degrees_spatial_v', 'GT_spatial_deg_product']
features_structure = ['pr_u', 'pr_v','ppr_u', 'ppr_v', 'lcc_u', 'lcc_v', 'and_u', 'and_v', 'dc_u', 'dc_v', 'katz_u', 'katz_v',
                              'sp', 'jc', 'aa', 'cn','pa', 'ra']
features_commu_inferee = ['louvain_density', 'same_louvain', 'infomap_density', 'same_infomap', 'sbm_density', 'same_sbm',
                            'leiden_density', 'same_leiden', "surprise_density", "same_surprise", 
                            "significance_density", "same_significance"]
features_embeddings = ["deepwalk_cos", "deepwalk_rank", "deepwalk_had_mean",  "deepwalk_had_std",
                        "n2v_homophily_cos", "n2v_homophily_rank", "n2v_homophily_had_mean",  "n2v_homophily_had_std",
                      "crosswalk_cos", "crosswalk_rank", "crosswalk_had_mean",  "crosswalk_had_std"]
all_matrices = {}
results_corr = []

for i in np.arange(0.00, 1.25, 0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv5_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    # 1. Chargement des données d'entraînement
    G_train, dataset_train, dataset_eval, _, _, _ = gp.load_all_data_for_graph(G_name)

    #features_to_test = [c for c in dataset_eval.columns if c not in features_GT and c not in ['u', 'v', 'target', 'label']]
    features_to_test = features_structure
    
    current_corr = dataset_eval[features_to_test + features_GT].corr(method='spearman')
    sub_matrix = current_corr.loc[features_to_test, features_GT]
    all_matrices[i] = sub_matrix

    #print(f"\n--- Matrice de Corrélation GT pour Alpha = {i} ---")
    print(sub_matrix.round(3))

    for f in features_to_test:
        for gt in features_GT:
            results_corr.append({
                'alpha': i,
                'feature': f,
                'gt_feature': gt,
                'correlation': current_corr.loc[f, gt]
            })

df_evolution = pd.DataFrame(results_corr)
features_GT = ['GT_sbm_density', 'GT_degrees_sbm_u', 'GT_degrees_sbm_v',
               'GT_pos_dist', 'GT_degrees_spatial_u','GT_degrees_spatial_v', 'GT_spatial_deg_product']
features_structure = ['pr_u', 'pr_v','ppr_u', 'ppr_v', 'lcc_u', 'lcc_v', 'and_u', 'and_v', 'dc_u', 'dc_v', 'katz_u', 'katz_v',
                              'sp', 'jc', 'aa', 'cn','pa', 'ra']
features_commu_inferee = ['louvain_density', 'same_louvain', 'infomap_density', 'same_infomap', 'sbm_density', 'same_sbm',
                            'leiden_density', 'same_leiden', "surprise_density", "same_surprise", 
                            "significance_density", "same_significance"]
features_embeddings = ["deepwalk_cos", "deepwalk_rank", "deepwalk_had_mean",  "deepwalk_had_std",
                        "n2v_homophily_cos", "n2v_homophily_rank", "n2v_homophily_had_mean",  "n2v_homophily_had_std",
                      "crosswalk_cos", "crosswalk_rank", "crosswalk_had_mean",  "crosswalk_had_std"]
all_matrices = {}
results_corr = []

for i in np.arange(0.00, 1.25, 0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv5_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    # 1. Chargement des données d'entraînement
    G_train, dataset_train, dataset_eval, _, _, _ = gp.load_all_data_for_graph(G_name)

    #features_to_test = [c for c in dataset_eval.columns if c not in features_GT and c not in ['u', 'v', 'target', 'label']]
    features_to_test = features_structure
    
    current_corr = dataset_eval[features_to_test + features_GT].corr(method='spearman')
    sub_matrix = current_corr.loc[features_to_test, features_GT]
    all_matrices[i] = sub_matrix

    #print(f"\n--- Matrice de Corrélation GT pour Alpha = {i} ---")
    print(sub_matrix.round(3))

    for f in features_to_test:
        for gt in features_GT:
            results_corr.append({
                'alpha': i,
                'feature': f,
                'gt_feature': gt,
                'correlation': current_corr.loc[f, gt]
            })

df_evolution = pd.DataFrame(results_corr)
gt_commu = 'same_GT_sbm'
gt_spatial = 'GT_pos_dist'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

def add_labels_inverted(ax, data, hue_col, x_target, alignment='left'):
    lines = ax.get_lines()
    features = data[hue_col].unique()
    for i, feature in enumerate(features):
        subset = data[data[hue_col] == feature]
        idx = (subset['alpha'] - x_target).abs().idxmin()
        point = subset.loc[idx]
        color = lines[i].get_color()
        offset = 0.02 if alignment == 'left' else -0.02
        ax.text(point['alpha'] + offset, point['correlation'], feature, fontsize=8, va='center', ha=alignment, color=color, fontweight='bold', alpha=0.7)

# --- Plot 1 : SBM (Labels à DROITE @ alpha=1.0) ---
data_commu = df_evolution[df_evolution['gt_feature'] == gt_commu]
sns.lineplot(data=data_commu, x='alpha', y='correlation', hue='feature', marker='o', ax=ax1)
ax1.set_title(f"Affinité Communautaire\n(Labels au régime Spatial $\\alpha=1$)", fontsize=13)
add_labels_inverted(ax1, data_commu, 'feature', x_target=1.0, alignment='left')
ax1.set_xlim(-0.05, 1.1) 
ax1.legend(title="Features", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='x-small', borderaxespad=0.)

# --- Plot 2 : Spatial (Labels à GAUCHE @ alpha=0.0) ---
data_spatial = df_evolution[df_evolution['gt_feature'] == gt_spatial]
sns.lineplot(data=data_spatial, x='alpha', y='correlation', hue='feature', marker='o', ax=ax2, legend=False)
ax2.set_title(f"Affinité Spatiale\n(Labels au régime SBM $\\alpha=0$)", fontsize=13)
add_labels_inverted(ax2, data_spatial, 'feature', x_target=0.0, alignment='right')
ax2.set_xlim(-0.1, 1.05)

for ax in [ax1, ax2]:
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlabel("Alpha (0=SBM, 1=Spatial)", fontweight='bold')
    ax.set_ylabel("Corrélation de Spearman", fontweight='bold')
    ax2.tick_params(labelleft=True)
    ax2.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    
plt.subplots_adjust(wspace=0.18) 
plt.show()
gt_commu = 'GT_sbm_density'
gt_spatial = 'GT_pos_dist'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

def add_labels_inverted(ax, data, hue_col, x_target, alignment='left'):
    lines = ax.get_lines()
    features = data[hue_col].unique()
    for i, feature in enumerate(features):
        subset = data[data[hue_col] == feature]
        idx = (subset['alpha'] - x_target).abs().idxmin()
        point = subset.loc[idx]
        color = lines[i].get_color()
        offset = 0.02 if alignment == 'left' else -0.02
        ax.text(point['alpha'] + offset, point['correlation'], feature, fontsize=8, va='center', ha=alignment, color=color, fontweight='bold', alpha=0.7)

# --- Plot 1 : SBM (Labels à DROITE @ alpha=1.0) ---
data_commu = df_evolution[df_evolution['gt_feature'] == gt_commu]
sns.lineplot(data=data_commu, x='alpha', y='correlation', hue='feature', marker='o', ax=ax1)
ax1.set_title(f"Affinité Communautaire\n(Labels au régime Spatial $\\alpha=1$)", fontsize=13)
add_labels_inverted(ax1, data_commu, 'feature', x_target=1.0, alignment='left')
ax1.set_xlim(-0.05, 1.1) 
ax1.legend(title="Features", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='x-small', borderaxespad=0.)

# --- Plot 2 : Spatial (Labels à GAUCHE @ alpha=0.0) ---
data_spatial = df_evolution[df_evolution['gt_feature'] == gt_spatial]
sns.lineplot(data=data_spatial, x='alpha', y='correlation', hue='feature', marker='o', ax=ax2, legend=False)
ax2.set_title(f"Affinité Spatiale\n(Labels au régime SBM $\\alpha=0$)", fontsize=13)
add_labels_inverted(ax2, data_spatial, 'feature', x_target=0.0, alignment='right')
ax2.set_xlim(-0.1, 1.05)

for ax in [ax1, ax2]:
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlabel("Alpha (0=SBM, 1=Spatial)", fontweight='bold')
    ax.set_ylabel("Corrélation de Spearman", fontweight='bold')
    ax2.tick_params(labelleft=True)
    ax2.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    
plt.subplots_adjust(wspace=0.18) 
plt.show()
features_GT = ['GT_sbm_density', 'GT_degrees_sbm_u', 'GT_degrees_sbm_v',
               'GT_pos_dist', 'GT_degrees_spatial_u','GT_degrees_spatial_v', 'GT_spatial_deg_product']
features_structure = ['pr_u', 'pr_v','ppr_u', 'ppr_v', 'lcc_u', 'lcc_v', 'and_u', 'and_v', 'dc_u', 'dc_v', 'katz_u', 'katz_v',
                              'sp', 'jc', 'aa', 'cn','pa', 'ra']
features_commu_inferee = ['louvain_density', 'same_louvain', 'infomap_density', 'same_infomap', 'sbm_density', 'same_sbm',
                            'leiden_density', 'same_leiden', "surprise_density", "same_surprise", 
                            "significance_density", "same_significance"]
features_embeddings = ["deepwalk_cos", "deepwalk_rank", "deepwalk_had_mean",  "deepwalk_had_std",
                        "n2v_homophily_cos", "n2v_homophily_rank", "n2v_homophily_had_mean",  "n2v_homophily_had_std",
                      "crosswalk_cos", "crosswalk_rank", "crosswalk_had_mean",  "crosswalk_had_std"]
all_matrices = {}
results_corr = []

for i in np.arange(0.00, 1.25, 0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv5_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    # 1. Chargement des données d'entraînement
    G_train, dataset_train, dataset_eval, _, _, _ = gp.load_all_data_for_graph(G_name)

    #features_to_test = [c for c in dataset_eval.columns if c not in features_GT and c not in ['u', 'v', 'target', 'label']]
    features_to_test = features_commu_inferee
    
    current_corr = dataset_eval[features_to_test + features_GT].corr(method='spearman')
    sub_matrix = current_corr.loc[features_to_test, features_GT]
    all_matrices[i] = sub_matrix

    #print(f"\n--- Matrice de Corrélation GT pour Alpha = {i} ---")
    print(sub_matrix.round(3))

    for f in features_to_test:
        for gt in features_GT:
            results_corr.append({
                'alpha': i,
                'feature': f,
                'gt_feature': gt,
                'correlation': current_corr.loc[f, gt]
            })

df_evolution = pd.DataFrame(results_corr)
gt_commu = 'GT_sbm_density'
gt_spatial = 'GT_pos_dist'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

def add_labels_inverted(ax, data, hue_col, x_target, alignment='left'):
    lines = ax.get_lines()
    features = data[hue_col].unique()
    for i, feature in enumerate(features):
        subset = data[data[hue_col] == feature]
        idx = (subset['alpha'] - x_target).abs().idxmin()
        point = subset.loc[idx]
        color = lines[i].get_color()
        offset = 0.02 if alignment == 'left' else -0.02
        ax.text(point['alpha'] + offset, point['correlation'], feature, fontsize=8, va='center', ha=alignment, color=color, fontweight='bold', alpha=0.7)

# --- Plot 1 : SBM (Labels à DROITE @ alpha=1.0) ---
data_commu = df_evolution[df_evolution['gt_feature'] == gt_commu]
sns.lineplot(data=data_commu, x='alpha', y='correlation', hue='feature', marker='o', ax=ax1)
ax1.set_title(f"Affinité Communautaire\n(Labels au régime Spatial $\\alpha=1$)", fontsize=13)
add_labels_inverted(ax1, data_commu, 'feature', x_target=1.0, alignment='left')
ax1.set_xlim(-0.05, 1.1) 
ax1.legend(title="Features", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='x-small', borderaxespad=0.)

# --- Plot 2 : Spatial (Labels à GAUCHE @ alpha=0.0) ---
data_spatial = df_evolution[df_evolution['gt_feature'] == gt_spatial]
sns.lineplot(data=data_spatial, x='alpha', y='correlation', hue='feature', marker='o', ax=ax2, legend=False)
ax2.set_title(f"Affinité Spatiale\n(Labels au régime SBM $\\alpha=0$)", fontsize=13)
add_labels_inverted(ax2, data_spatial, 'feature', x_target=0.0, alignment='right')
ax2.set_xlim(-0.1, 1.05)

for ax in [ax1, ax2]:
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlabel("Alpha (0=SBM, 1=Spatial)", fontweight='bold')
    ax.set_ylabel("Corrélation de Spearman", fontweight='bold')
    ax2.tick_params(labelleft=True)
    ax2.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    
plt.subplots_adjust(wspace=0.18) 
plt.show()
features_GT = ['GT_sbm_density', 'GT_degrees_sbm_u', 'GT_degrees_sbm_v',
               'GT_pos_dist', 'GT_degrees_spatial_u','GT_degrees_spatial_v', 'GT_spatial_deg_product']
features_structure = ['pr_u', 'pr_v','ppr_u', 'ppr_v', 'lcc_u', 'lcc_v', 'and_u', 'and_v', 'dc_u', 'dc_v', 'katz_u', 'katz_v',
                              'sp', 'jc', 'aa', 'cn','pa', 'ra']
features_commu_inferee = ['louvain_density', 'same_louvain', 'infomap_density', 'same_infomap', 'sbm_density', 'same_sbm',
                            'leiden_density', 'same_leiden', "surprise_density", "same_surprise", 
                            "significance_density", "same_significance"]
features_embeddings = ["deepwalk_cos", "deepwalk_rank", "deepwalk_had_mean",  "deepwalk_had_std",
                        "n2v_homophily_cos", "n2v_homophily_rank", "n2v_homophily_had_mean",  "n2v_homophily_had_std",
                      "crosswalk_cos", "crosswalk_rank", "crosswalk_had_mean",  "crosswalk_had_std"]
all_matrices = {}
results_corr = []

for i in np.arange(0.00, 1.25, 0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv5_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    # 1. Chargement des données d'entraînement
    G_train, dataset_train, dataset_eval, _, _, _ = gp.load_all_data_for_graph(G_name)

    #features_to_test = [c for c in dataset_eval.columns if c not in features_GT and c not in ['u', 'v', 'target', 'label']]
    features_to_test = features_embeddings
    
    current_corr = dataset_eval[features_to_test + features_GT].corr(method='spearman')
    sub_matrix = current_corr.loc[features_to_test, features_GT]
    all_matrices[i] = sub_matrix

    #print(f"\n--- Matrice de Corrélation GT pour Alpha = {i} ---")
    print(sub_matrix.round(3))

    for f in features_to_test:
        for gt in features_GT:
            results_corr.append({
                'alpha': i,
                'feature': f,
                'gt_feature': gt,
                'correlation': current_corr.loc[f, gt]
            })

df_evolution = pd.DataFrame(results_corr)
gt_commu = 'GT_sbm_density'
gt_spatial = 'GT_pos_dist'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

def add_labels_inverted(ax, data, hue_col, x_target, alignment='left'):
    lines = ax.get_lines()
    features = data[hue_col].unique()
    for i, feature in enumerate(features):
        subset = data[data[hue_col] == feature]
        idx = (subset['alpha'] - x_target).abs().idxmin()
        point = subset.loc[idx]
        color = lines[i].get_color()
        offset = 0.02 if alignment == 'left' else -0.02
        ax.text(point['alpha'] + offset, point['correlation'], feature, fontsize=8, va='center', ha=alignment, color=color, fontweight='bold', alpha=0.7)

# --- Plot 1 : SBM (Labels à DROITE @ alpha=1.0) ---
data_commu = df_evolution[df_evolution['gt_feature'] == gt_commu]
sns.lineplot(data=data_commu, x='alpha', y='correlation', hue='feature', marker='o', ax=ax1)
ax1.set_title(f"Affinité Communautaire\n(Labels au régime Spatial $\\alpha=1$)", fontsize=13)
add_labels_inverted(ax1, data_commu, 'feature', x_target=1.0, alignment='left')
ax1.set_xlim(-0.05, 1.1) 
ax1.legend(title="Features", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='x-small', borderaxespad=0.)

# --- Plot 2 : Spatial (Labels à GAUCHE @ alpha=0.0) ---
data_spatial = df_evolution[df_evolution['gt_feature'] == gt_spatial]
sns.lineplot(data=data_spatial, x='alpha', y='correlation', hue='feature', marker='o', ax=ax2, legend=False)
ax2.set_title(f"Affinité Spatiale\n(Labels au régime SBM $\\alpha=0$)", fontsize=13)
add_labels_inverted(ax2, data_spatial, 'feature', x_target=0.0, alignment='right')
ax2.set_xlim(-0.1, 1.05)

for ax in [ax1, ax2]:
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlabel("Alpha (0=SBM, 1=Spatial)", fontweight='bold')
    ax.set_ylabel("Corrélation de Spearman", fontweight='bold')
    ax2.tick_params(labelleft=True)
    ax2.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    
plt.subplots_adjust(wspace=0.18) 
plt.show()
features_GT = ['GT_sbm_density', 'GT_degrees_sbm_u', 'GT_degrees_sbm_v',
               'GT_pos_dist', 'GT_degrees_spatial_u','GT_degrees_spatial_v', 'GT_spatial_deg_product']
features_structure = ['pr_u', 'pr_v','ppr_u', 'ppr_v', 'lcc_u', 'lcc_v', 'and_u', 'and_v', 'dc_u', 'dc_v', 'katz_u', 'katz_v',
                              'sp', 'jc', 'aa', 'cn','pa', 'ra']
features_commu_inferee = ['louvain_density', 'same_louvain', 'infomap_density', 'same_infomap', 'sbm_density', 'same_sbm',
                            'leiden_density', 'same_leiden', "surprise_density", "same_surprise", 
                            "significance_density", "same_significance"]
features_embeddings = ["deepwalk_cos", "deepwalk_rank", "deepwalk_had_mean",  "deepwalk_had_std",
                        "n2v_homophily_cos", "n2v_homophily_rank", "n2v_homophily_had_mean",  "n2v_homophily_had_std",
                      "crosswalk_cos", "crosswalk_rank", "crosswalk_had_mean",  "crosswalk_had_std"]
all_matrices = {}
results_corr = []

for i in np.arange(0.00, 1.25, 0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv5_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    # 1. Chargement des données d'entraînement
    G_train, dataset_train, dataset_eval, _, _, _ = gp.load_all_data_for_graph(G_name)

    #features_to_test = [c for c in dataset_eval.columns if c not in features_GT and c not in ['u', 'v', 'target', 'label']]
    features_to_test = features_commu_inferee
    
    current_corr = dataset_eval[features_to_test + features_GT].corr(method='spearman')
    sub_matrix = current_corr.loc[features_to_test, features_GT]
    all_matrices[i] = sub_matrix

    #print(f"\n--- Matrice de Corrélation GT pour Alpha = {i} ---")
    print(sub_matrix.round(3))

    for f in features_to_test:
        for gt in features_GT:
            results_corr.append({
                'alpha': i,
                'feature': f,
                'gt_feature': gt,
                'correlation': current_corr.loc[f, gt]
            })

df_evolution = pd.DataFrame(results_corr)
gt_commu = 'GT_sbm_density'
gt_spatial = 'GT_pos_dist'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

def add_labels_inverted(ax, data, hue_col, x_target, alignment='left'):
    lines = ax.get_lines()
    features = data[hue_col].unique()
    for i, feature in enumerate(features):
        subset = data[data[hue_col] == feature]
        idx = (subset['alpha'] - x_target).abs().idxmin()
        point = subset.loc[idx]
        color = lines[i].get_color()
        offset = 0.02 if alignment == 'left' else -0.02
        ax.text(point['alpha'] + offset, point['correlation'], feature, fontsize=8, va='center', ha=alignment, color=color, fontweight='bold', alpha=0.7)

# --- Plot 1 : SBM (Labels à DROITE @ alpha=1.0) ---
data_commu = df_evolution[df_evolution['gt_feature'] == gt_commu]
sns.lineplot(data=data_commu, x='alpha', y='correlation', hue='feature', marker='o', ax=ax1)
ax1.set_title(f"Affinité Communautaire\n(Labels au régime Spatial $\\alpha=1$)", fontsize=13)
add_labels_inverted(ax1, data_commu, 'feature', x_target=1.0, alignment='left')
ax1.set_xlim(-0.05, 1.1) 
ax1.legend(title="Features", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='x-small', borderaxespad=0.)

# --- Plot 2 : Spatial (Labels à GAUCHE @ alpha=0.0) ---
data_spatial = df_evolution[df_evolution['gt_feature'] == gt_spatial]
sns.lineplot(data=data_spatial, x='alpha', y='correlation', hue='feature', marker='o', ax=ax2, legend=False)
ax2.set_title(f"Affinité Spatiale\n(Labels au régime SBM $\\alpha=0$)", fontsize=13)
add_labels_inverted(ax2, data_spatial, 'feature', x_target=0.0, alignment='right')
ax2.set_xlim(-0.1, 1.05)

for ax in [ax1, ax2]:
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlabel("Alpha (0=SBM, 1=Spatial)", fontweight='bold')
    ax.set_ylabel("Corrélation de Spearman", fontweight='bold')
    ax2.tick_params(labelleft=True)
    ax2.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    
plt.subplots_adjust(wspace=0.18) 
plt.show()
features_GT = ['GT_sbm_density', 'GT_degrees_sbm_u', 'GT_degrees_sbm_v',
               'GT_pos_dist', 'GT_degrees_spatial_u','GT_degrees_spatial_v', 'GT_spatial_deg_product']
features_structure = ['pr_u', 'pr_v','ppr_u', 'ppr_v', 'lcc_u', 'lcc_v', 'and_u', 'and_v', 'dc_u', 'dc_v', 'katz_u', 'katz_v',
                              'sp', 'jc', 'aa', 'cn','pa', 'ra']
features_commu_inferee = ['louvain_density', 'same_louvain', 'infomap_density', 'same_infomap', 'sbm_density', 'same_sbm',
                            'leiden_density', 'same_leiden', "surprise_density", "same_surprise", 
                            "significance_density", "same_significance"]
features_embeddings = ["deepwalk_cos", "deepwalk_rank", "deepwalk_had_mean",  "deepwalk_had_std",
                        "n2v_homophily_cos", "n2v_homophily_rank", "n2v_homophily_had_mean",  "n2v_homophily_had_std",
                      "crosswalk_cos", "crosswalk_rank", "crosswalk_had_mean",  "crosswalk_had_std"]
all_matrices = {}
results_corr = []

for i in np.arange(0.00, 1.25, 0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv5_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    # 1. Chargement des données d'entraînement
    G_train, dataset_train, dataset_eval, _, _, _ = gp.load_all_data_for_graph(G_name)

    #features_to_test = [c for c in dataset_eval.columns if c not in features_GT and c not in ['u', 'v', 'target', 'label']]
    features_to_test = features_embeddings
    
    current_corr = dataset_eval[features_to_test + features_GT].corr(method='spearman')
    sub_matrix = current_corr.loc[features_to_test, features_GT]
    all_matrices[i] = sub_matrix

    #print(f"\n--- Matrice de Corrélation GT pour Alpha = {i} ---")
    print(sub_matrix.round(3))

    for f in features_to_test:
        for gt in features_GT:
            results_corr.append({
                'alpha': i,
                'feature': f,
                'gt_feature': gt,
                'correlation': current_corr.loc[f, gt]
            })

df_evolution = pd.DataFrame(results_corr)
gt_commu = 'GT_sbm_density'
gt_spatial = 'GT_pos_dist'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

def add_labels_inverted(ax, data, hue_col, x_target, alignment='left'):
    lines = ax.get_lines()
    features = data[hue_col].unique()
    for i, feature in enumerate(features):
        subset = data[data[hue_col] == feature]
        idx = (subset['alpha'] - x_target).abs().idxmin()
        point = subset.loc[idx]
        color = lines[i].get_color()
        offset = 0.02 if alignment == 'left' else -0.02
        ax.text(point['alpha'] + offset, point['correlation'], feature, fontsize=8, va='center', ha=alignment, color=color, fontweight='bold', alpha=0.7)

# --- Plot 1 : SBM (Labels à DROITE @ alpha=1.0) ---
data_commu = df_evolution[df_evolution['gt_feature'] == gt_commu]
sns.lineplot(data=data_commu, x='alpha', y='correlation', hue='feature', marker='o', ax=ax1)
ax1.set_title(f"Affinité Communautaire\n(Labels au régime Spatial $\\alpha=1$)", fontsize=13)
add_labels_inverted(ax1, data_commu, 'feature', x_target=1.0, alignment='left')
ax1.set_xlim(-0.05, 1.1) 
ax1.legend(title="Features", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='x-small', borderaxespad=0.)

# --- Plot 2 : Spatial (Labels à GAUCHE @ alpha=0.0) ---
data_spatial = df_evolution[df_evolution['gt_feature'] == gt_spatial]
sns.lineplot(data=data_spatial, x='alpha', y='correlation', hue='feature', marker='o', ax=ax2, legend=False)
ax2.set_title(f"Affinité Spatiale\n(Labels au régime SBM $\\alpha=0$)", fontsize=13)
add_labels_inverted(ax2, data_spatial, 'feature', x_target=0.0, alignment='right')
ax2.set_xlim(-0.1, 1.05)

for ax in [ax1, ax2]:
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlabel("Alpha (0=SBM, 1=Spatial)", fontweight='bold')
    ax.set_ylabel("Corrélation de Spearman", fontweight='bold')
    ax2.tick_params(labelleft=True)
    ax2.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    
plt.subplots_adjust(wspace=0.18) 
plt.show()
features_GT = ['GT_sbm_density', 'GT_degrees_sbm_u', 'GT_degrees_sbm_v',
               'GT_pos_dist', 'GT_degrees_spatial_u','GT_degrees_spatial_v', 'GT_spatial_deg_product']
features_structure = ['pr_u', 'pr_v','ppr_u', 'ppr_v', 'lcc_u', 'lcc_v', 'and_u', 'and_v', 'dc_u', 'dc_v', 'katz_u', 'katz_v',
                              'sp', 'jc', 'aa', 'cn','pa', 'ra']
features_commu_inferee = ['louvain_density', 'same_louvain', 'infomap_density', 'same_infomap', 'sbm_density', 'same_sbm',
                            'leiden_density', 'same_leiden', "surprise_density", "same_surprise", 
                            "significance_density", "same_significance"]
features_embeddings = ["deepwalk_cos", "deepwalk_rank", "deepwalk_had_mean",  "deepwalk_had_std",
                        "n2v_homophily_cos", "n2v_homophily_rank", "n2v_homophily_had_mean",  "n2v_homophily_had_std",
                      "crosswalk_cos", "crosswalk_rank", "crosswalk_had_mean",  "crosswalk_had_std"]
all_matrices = {}
results_corr = []

for i in np.arange(0.00, 1.25, 0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv5_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    # 1. Chargement des données d'entraînement
    G_train, dataset_train, dataset_eval, _, _, _ = gp.load_all_data_for_graph(G_name)

    #features_to_test = [c for c in dataset_eval.columns if c not in features_GT and c not in ['u', 'v', 'target', 'label']]
    features_to_test = features_structure
    
    current_corr = dataset_eval[features_to_test + features_GT].corr(method='spearman')
    sub_matrix = current_corr.loc[features_to_test, features_GT]
    all_matrices[i] = sub_matrix

    #print(f"\n--- Matrice de Corrélation GT pour Alpha = {i} ---")
    print(sub_matrix.round(3))

    for f in features_to_test:
        for gt in features_GT:
            results_corr.append({
                'alpha': i,
                'feature': f,
                'gt_feature': gt,
                'correlation': current_corr.loc[f, gt]
            })

df_evolution = pd.DataFrame(results_corr)
gt_commu = 'GT_sbm_density'
gt_spatial = 'GT_pos_dist'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

def add_labels_inverted(ax, data, hue_col, x_target, alignment='left'):
    lines = ax.get_lines()
    features = data[hue_col].unique()
    for i, feature in enumerate(features):
        subset = data[data[hue_col] == feature]
        idx = (subset['alpha'] - x_target).abs().idxmin()
        point = subset.loc[idx]
        color = lines[i].get_color()
        offset = 0.02 if alignment == 'left' else -0.02
        ax.text(point['alpha'] + offset, point['correlation'], feature, fontsize=8, va='center', ha=alignment, color=color, fontweight='bold', alpha=0.7)

# --- Plot 1 : SBM (Labels à DROITE @ alpha=1.0) ---
data_commu = df_evolution[df_evolution['gt_feature'] == gt_commu]
sns.lineplot(data=data_commu, x='alpha', y='correlation', hue='feature', marker='o', ax=ax1)
ax1.set_title(f"Affinité Communautaire\n(Labels au régime Spatial $\\alpha=1$)", fontsize=13)
add_labels_inverted(ax1, data_commu, 'feature', x_target=1.0, alignment='left')
ax1.set_xlim(-0.05, 1.1) 
ax1.legend(title="Features", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='x-small', borderaxespad=0.)

# --- Plot 2 : Spatial (Labels à GAUCHE @ alpha=0.0) ---
data_spatial = df_evolution[df_evolution['gt_feature'] == gt_spatial]
sns.lineplot(data=data_spatial, x='alpha', y='correlation', hue='feature', marker='o', ax=ax2, legend=False)
ax2.set_title(f"Affinité Spatiale\n(Labels au régime SBM $\\alpha=0$)", fontsize=13)
add_labels_inverted(ax2, data_spatial, 'feature', x_target=0.0, alignment='right')
ax2.set_xlim(-0.1, 1.05)

for ax in [ax1, ax2]:
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlabel("Alpha (0=SBM, 1=Spatial)", fontweight='bold')
    ax.set_ylabel("Corrélation de Spearman", fontweight='bold')
    ax2.tick_params(labelleft=True)
    ax2.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    
plt.subplots_adjust(wspace=0.18) 
plt.show()
all_summaries = []

for i in np.arange(0, 1.25, 0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv4_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}_GT"
    
    _, _, _, xgboost_data, shap_explainer, _ = gp.load_all_data_for_graph(G_name)
    X_test = xgboost_data['X_test']

    print(f"Type de l'objet : {type(shap_explainer)}")
    print(f"Dimensions des valeurs : {shap_explainer.values.shape}")

    vals = shap_explainer.values
    
    if isinstance(vals, list):
        vals = vals[1]

    df_shap = pd.DataFrame(vals, columns=X_test.columns)
    mean_shap = df_shap.mean()
    mean_abs_shap = df_shap.abs().mean()
    
    summary = pd.DataFrame({
        'feature': X_test.columns,
        'mean_shap': mean_shap.values,
        'mean_abs_shap': mean_abs_shap.values,
        'sbm_param': i
    })
    all_summaries.append(summary)

df_shapvals_concat = pd.concat(all_summaries)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_importance_evolution(df_results, metric="mean_shap", as_ratio=True):
    """
    Génère une heatmap montrant l'évolution de l'importance des features (SHAP).
    """
    heatmap_data = df_results.pivot(index='feature', columns='sbm_param', values=metric)
    
    if as_ratio:
        heatmap_data = heatmap_data.div(heatmap_data.sum(axis=0), axis=1)
        label_name = f"Ratio de contribution ({metric})"
        fmt_type = ".2%"
    else:
        label_name = f"Valeur brute ({metric})"
        fmt_type = ".3f"

    heatmap_data['avg_importance'] = heatmap_data.mean(axis=1)
    heatmap_data = heatmap_data.sort_values(by='avg_importance', ascending=False).drop(columns='avg_importance')

    plt.figure(figsize=(16, 9))
    sns.heatmap(heatmap_data, annot=True, fmt=fmt_type, cmap="YlGnBu", yticklabels=True, cbar_kws={'label': label_name})

    title_suffix = "(Ratios normalisés)" if as_ratio else "(Valeurs brutes)"
    plt.title(f"Importance relative des features (SHAP) selon la structure du graphe {title_suffix}")
    plt.xlabel("Paramètre du graphe (i)")
    plt.ylabel("Features")
    plt.tight_layout()

    filename = f"outputs/plots/heatmap_feat_importance_per_ratio_v4_{metric}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé sous : {filename}")
    plt.show()

plot_feature_importance_evolution(df_shapvals_concat, metric="mean_abs_shap", as_ratio=False)
print(df_shapvals_concat.columns)
features_list = df_shapvals_concat['feature']
print(", ".join(map(str, features_list)))

"""
all_distribs = []
for i in np.arange(0, 1.05, 0.05):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbm_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    _, _,_, xgboost_data, _, _ = gp.load_all_data_for_graph(G_name)
    X_test = xgboost_data['X_test']
    y_test = xgboost_data['y_test']
    
    pos_sp = X_test[y_test == 1]['sp'].value_counts(normalize=True).to_dict()
    neg_sp = X_test[y_test == 0]['sp'].value_counts(normalize=True).to_dict()
    
    row = {'sbm_param': round(i, 2)}
    for dist in range(1, 7):
        row[f'pos_sp_{dist}'] = pos_sp.get(dist, 0)
        row[f'neg_sp_{dist}'] = neg_sp.get(dist, 0)
    
    row['pos_sp_7+'] = sum(v for k, v in pos_sp.items() if k >= 7 or k == 0)
    row['neg_sp_7+'] = sum(v for k, v in neg_sp.items() if k >= 7 or k == 0)
    
    all_distribs.append(row)

df_sp_detailed = pd.DataFrame(all_distribs)

cols = ['sbm_param'] + [f'{prefix}_sp_{d}' for d in list(range(1, 7)) + ['7+'] for prefix in ['pos', 'neg']]
df_sp_detailed = df_sp_detailed[cols]

pd.options.display.float_format = '{:.3f}'.format
print(df_sp_detailed.to_string(index=False))
"""
all_summaries = []

for i in np.arange(0, 1.25, 0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv5_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}_GT"
    
    _, _, _, xgboost_data, shap_explainer, _ = gp.load_all_data_for_graph(G_name)
    X_test = xgboost_data['X_test']

    print(f"Type de l'objet : {type(shap_explainer)}")
    print(f"Dimensions des valeurs : {shap_explainer.values.shape}")

    vals = shap_explainer.values
    
    if isinstance(vals, list):
        vals = vals[1]

    df_shap = pd.DataFrame(vals, columns=X_test.columns)
    mean_shap = df_shap.mean()
    mean_abs_shap = df_shap.abs().mean()
    
    summary = pd.DataFrame({
        'feature': X_test.columns,
        'mean_shap': mean_shap.values,
        'mean_abs_shap': mean_abs_shap.values,
        'sbm_param': i
    })
    all_summaries.append(summary)

df_shapvals_concat = pd.concat(all_summaries)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_importance_evolution(df_results, metric="mean_shap", as_ratio=True):
    """
    Génère une heatmap montrant l'évolution de l'importance des features (SHAP).
    """
    heatmap_data = df_results.pivot(index='feature', columns='sbm_param', values=metric)
    
    if as_ratio:
        heatmap_data = heatmap_data.div(heatmap_data.sum(axis=0), axis=1)
        label_name = f"Ratio de contribution ({metric})"
        fmt_type = ".2%"
    else:
        label_name = f"Valeur brute ({metric})"
        fmt_type = ".3f"

    heatmap_data['avg_importance'] = heatmap_data.mean(axis=1)
    heatmap_data = heatmap_data.sort_values(by='avg_importance', ascending=False).drop(columns='avg_importance')

    plt.figure(figsize=(16, 9))
    sns.heatmap(heatmap_data, annot=True, fmt=fmt_type, cmap="YlGnBu", yticklabels=True, cbar_kws={'label': label_name})

    title_suffix = "(Ratios normalisés)" if as_ratio else "(Valeurs brutes)"
    plt.title(f"Importance relative des features (SHAP) selon la structure du graphe {title_suffix}")
    plt.xlabel("Paramètre du graphe (i)")
    plt.ylabel("Features")
    plt.tight_layout()

    filename = f"outputs/plots/heatmap_feat_importance_per_ratio_v4_{metric}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé sous : {filename}")
    plt.show()

plot_feature_importance_evolution(df_shapvals_concat, metric="mean_abs_shap", as_ratio=False)
print(df_shapvals_concat.columns)
features_list = df_shapvals_concat['feature']
print(", ".join(map(str, features_list)))

"""
all_distribs = []
for i in np.arange(0, 1.05, 0.05):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbm_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    _, _,_, xgboost_data, _, _ = gp.load_all_data_for_graph(G_name)
    X_test = xgboost_data['X_test']
    y_test = xgboost_data['y_test']
    
    pos_sp = X_test[y_test == 1]['sp'].value_counts(normalize=True).to_dict()
    neg_sp = X_test[y_test == 0]['sp'].value_counts(normalize=True).to_dict()
    
    row = {'sbm_param': round(i, 2)}
    for dist in range(1, 7):
        row[f'pos_sp_{dist}'] = pos_sp.get(dist, 0)
        row[f'neg_sp_{dist}'] = neg_sp.get(dist, 0)
    
    row['pos_sp_7+'] = sum(v for k, v in pos_sp.items() if k >= 7 or k == 0)
    row['neg_sp_7+'] = sum(v for k, v in neg_sp.items() if k >= 7 or k == 0)
    
    all_distribs.append(row)

df_sp_detailed = pd.DataFrame(all_distribs)

cols = ['sbm_param'] + [f'{prefix}_sp_{d}' for d in list(range(1, 7)) + ['7+'] for prefix in ['pos', 'neg']]
df_sp_detailed = df_sp_detailed[cols]

pd.options.display.float_format = '{:.3f}'.format
print(df_sp_detailed.to_string(index=False))
"""
# On garde tes groupes pour l'analyse
group_mapping = {
    "Groupe_Structure": ['cn', 'aa', 'jc', 'pa', 'sp', 'pr_u', 'pr_v', 'lcc_u', 'lcc_v', 'and_u', 'and_v', 'dc_u', 'dc_v'],
    "Groupe_Communities": ['sbm_density', 'same_sbm', 'infomap_density', 'same_infomap',"louvain_density", "same_louvain"],
    "Groupe_Embeddings": ['n2v_homophily_cos', 'n2v_homophily_dist', 'deepwalk_cos', 'deepwalk_dist']
}

output_folder = "outputs/plots"
os.makedirs(output_folder, exist_ok=True)

ratios = [0, 0.25, 0.50, 0.75, 1.00]
for i in ratios:
    print(f"RRRRRRRRRRRRATIO i : {i}")
    sbm_val_str = f"{i:.2f}"
    pos_val_str = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv2_{sbm_val_str.replace('.', '_')}_pos_{pos_val_str.replace('.', '_')}"
    
    _, dataset_train, _, _, _, _ = gp.load_all_data_for_graph(G_name)    
    features = [c for c in dataset_train.columns if c != 'target']
    df_features = dataset_train[features]

    print(f"Features ac 1 unique valeur pour rati_sbm={sbm_val_str} : {df_features.loc[:, df_features.nunique() <= 1].head(1)}")
    df_features = df_features.loc[:, df_features.nunique() > 1]

    # --- CALCUL DE LA MATRICE DE CORRÉLATION ---
    corr_matrix = df_features.corr(method='pearson')
    
    # --- PLOT 1 : CLUSTERMAP (La vue d'ensemble) ---
    # Combine la heatmap et le dendrogramme
    g = sns.clustermap(corr_matrix, 
                       method='complete', 
                       cmap='RdBu_r', 
                       vmin=-1, vmax=1,
                       figsize=(12, 10))
    g.fig.suptitle(f"Clusters d'inter-corrélation (Ratio SBM: {sbm_val_str})", y=1.02)
    
    save_path = os.path.join(output_folder, f"correlation_clustermap_sbm_{sbm_val_str.replace('.', '_')}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Graphique sauvegardé : {save_path}")

    plt.show()

    # --- PLOT 2 : DENDROGRAMME PUR (Focus structure) ---
    plt.figure(figsize=(10, 5))
    features_clean = df_features.columns.tolist()
    # Distance = 1 - valeur absolue de la corrélation
    dissimilarity = 1 - np.abs(corr_matrix)
    # Linkage sur la matrice de corrélation nettoyée
    Z = hierarchy.linkage(squareform(dissimilarity, checks=False), method='complete')
    
    hierarchy.dendrogram(Z, labels=features_clean, leaf_rotation=90, color_threshold=0.3)
    plt.axhline(y=0.3, color='r', linestyle='--', label='Seuil de forte corrélation (0.7)')
    plt.title(f"Hiérarchie des variables (Ratio SBM: {sbm_val_str})")
    plt.ylabel("Dissimilarité (1 - |corr|)")
    plt.legend()
    plt.tight_layout()
    plt.show()
all_top_features = {}

for i in np.arange(1.00, -0.25, -0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    ratio_label = f"SBM_{sbm_val}"
    G_name = f"artificial_graph_sbmv5_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    _, _, _, xgboost_data, _, _ = gp.load_all_data_for_graph(G_name)  
    model = xgboost_data['model']

    importance_gain = model.get_booster().get_score(importance_type='gain')
    df_importance = pd.DataFrame(
        list(importance_gain.items()), 
        columns=['Feature', 'Gain']
    ).sort_values(by='Gain', ascending=False)
    
    total_gain = df_importance['Gain'].sum()
    df_importance['Weight_pct'] = (df_importance['Gain'] / total_gain) * 100
    
    all_top_features[ratio_label] = df_importance.head(20).reset_index(drop=True)

# --- Affichage pour analyse ---
summary_df = pd.DataFrame({
    ratio: all_top_features[ratio]['Feature'] for ratio in all_top_features.keys()
})

print("### Top 20 Features par Ratio (Evolution du Rang) ###")
print(summary_df)

# Optionnel : Sauvegarde pour analyse externe
# summary_df.to_csv("feature_rank_evolution.csv")
rank_dfs = {}

for i in [1.0, 0.75, 0.5, 0.25, 0.0]:
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    ratio_label = f"Ratio_SBM_{sbm_val}"
    G_name = f"artificial_graph_sbmv5_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    G_train, dataset_train, dataset_eval, xgboost_data, _, _ = gp.load_all_data_for_graph(G_name)
    model = xgboost_data['model']
    importance_gain = model.get_booster().get_score(importance_type='gain')
    
    df = pd.DataFrame(list(importance_gain.items()), columns=['Feature', 'Gain'])
    df = df.sort_values(by='Gain', ascending=False).reset_index(drop=True)
    
    df['Rank'] = df.index + 1
    df['Weight_pct'] = (df['Gain'] / df['Gain'].sum()) * 100
    
    rank_dfs[ratio_label] = df

# --- Analyse Comparative ---
# 1. On crée une table pivot des rangs pour voir l'évolution
all_features = set().union(*[set(df['Feature']) for df in rank_dfs.values()])
comparison_table = pd.DataFrame({'Feature': list(all_features)})

for label, df in rank_dfs.items():
    comparison_table = comparison_table.merge(
        df[['Feature', 'Rank']].rename(columns={'Rank': label}),
        on='Feature', how='left'
    )

# 2. Calcul du "Shift" (Dérive de rang) entre Full SBM (1.0) et Full Spatial (0.0)
comparison_table['Shift_SBM_vs_Spatial'] = comparison_table['Ratio_SBM_0.00'] - comparison_table['Ratio_SBM_1.00']

print("### TOP 10 FEATURES : SBM (1.0) vs SPATIAL (0.0) ###")
top_sbm = rank_dfs['Ratio_SBM_1.00'].head(10)[['Feature', 'Weight_pct']]
top_spatial = rank_dfs['Ratio_SBM_0.00'].head(10)[['Feature', 'Weight_pct']]

print("\n--- TOP SBM 1.0 ---")
print(top_sbm)
print("\n--- TOP SPATIAL 0.0 ---")
print(top_spatial)

# 3. Identification des "Spécialistes"
print("\n### FEATURES SPÉCIALISTES (Fort Shift de Rang) ###")
# On filtre celles qui sont au moins dans le Top 30 quelque part pour éviter le bruit du bas de tableau
specialists = comparison_table[
    (comparison_table['Ratio_SBM_1.00'] < 30) | (comparison_table['Ratio_SBM_0.00'] < 30)
].sort_values(by='Shift_SBM_vs_Spatial', ascending=False)

print("\nMarqueurs du monde SBM (Perdent beaucoup de rang en Spatial) :")
print(specialists.head(5)[['Feature', 'Shift_SBM_vs_Spatial']])

print("\nMarqueurs du monde Spatial (Perdent beaucoup de rang en SBM) :")
print(specialists.tail(5)[['Feature', 'Shift_SBM_vs_Spatial']])
rank_dfs = {}

for i in [1.0, 0.75, 0.5, 0.25, 0.0]:
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    ratio_label = f"Ratio_SBM_{sbm_val}"
    G_name = f"artificial_graph_sbmv5_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    G_train, dataset_train, dataset_eval, xgboost_data, _, _ = gp.load_all_data_for_graph(G_name)
    model = xgboost_data['model']
    importance_gain = model.get_booster().get_score(importance_type='gain')
    
    df = pd.DataFrame(list(importance_gain.items()), columns=['Feature', 'Gain'])
    df = df.sort_values(by='Gain', ascending=False).reset_index(drop=True)
    
    df['Rank'] = df.index + 1
    df['Weight_pct'] = (df['Gain'] / df['Gain'].sum()) * 100
    
    rank_dfs[ratio_label] = df

# --- 1. PRÉPARATION DES DONNÉES ---
if "Ratio_SBM_1.00" in rank_dfs and "Ratio_SBM_0.00" in rank_dfs:
    df_100 = rank_dfs["Ratio_SBM_1.00"]
    df_000 = rank_dfs["Ratio_SBM_0.00"]
    
    # Merge
    comparison = pd.merge(
        df_100, df_000, on='Feature', 
        suffixes='_SBM', '_Spatial', how='outer'
    )
    
    # --- 2. TRAITEMENT DES FEATURES ABSENTES (Anti-NaN) ---
    # Si une feature est absente, son poids est 0 et son rang est le max (ex: 100)
    comparison['W_pct_SBM'] = comparison['W_pct_SBM'].fillna(0)
    comparison['W_pct_Spatial'] = comparison['W_pct_Spatial'].fillna(0)
    comparison['Gain_Raw_SBM'] = comparison['Gain_Raw_SBM'].fillna(0)
    comparison['Gain_Raw_Spatial'] = comparison['Gain_Raw_Spatial'].fillna(0)
    
    # Pour le rang, on met une valeur arbitraire haute (ex: 100) pour marquer l'absence
    comparison['Rank_SBM'] = comparison['Rank_SBM'].fillna(100)
    comparison['Rank_Spatial'] = comparison['Rank_Spatial'].fillna(100)
    
    # --- 3. CALCUL DES MÉTRIQUES D'ANALYSE ---
    comparison['Rank_Shift'] = comparison['Rank_Spatial'] - comparison['Rank_SBM']
    
    # On calcule la différence de poids relatif (très parlant pour l'explainability)
    comparison['Weight_Delta'] = comparison['W_pct_SBM'] - comparison['W_pct_Spatial']

    # --- 4. FILTRAGE DES "VRAIS" ACTEURS ---
    # On ne s'intéresse qu'aux features qui "existent" significativement (Top 15 d'un côté ou de l'autre)
    top_features_mask = (comparison['Rank_SBM'] <= 15) | (comparison['Rank_Spatial'] <= 15)
    relevant_comparison = comparison[top_features_mask].copy()

    # Affichage
    pd.options.display.float_format = '{:.2f}'.format
    cols = ['Feature', 'Rank_SBM', 'W_pct_SBM', 'Rank_Spatial', 'W_pct_Spatial', 'Rank_Shift', 'Weight_Delta']

    print("\n" + "="*90)
    print("🎯 ANALYSE DES ACTEURS MAJEURS (TOP 15 SBM OU SPATIAL)")
    print("="*90)

    print("\n🔥 MARQUEURS SBM (Dominent le SBM, chutent en Spatial) :")
    # Tri par Weight_Delta positif (poids SBM >> poids Spatial)
    print(relevant_comparison.sort_values('Weight_Delta', ascending=False).head(10)[cols])

    print("\n🌍 MARQUEURS SPATIAUX (Dominent le Spatial, chutent en SBM) :")
    # Tri par Weight_Delta négatif (poids Spatial >> poids SBM)
    print(relevant_comparison.sort_values('Weight_Delta', ascending=True).head(10)[cols])

    print("\n🧱 PILIERS STRUCTURELS (Importants partout) :")
    # Features dans le top 15 des deux côtés avec un Delta faible
    piliers = relevant_comparison[
        (relevant_comparison['Rank_SBM'] <= 15) & 
        (relevant_comparison['Rank_Spatial'] <= 15) &
        (relevant_comparison['Weight_Delta'].abs() < 5)
    ]
    print(piliers.sort_values('Rank_SBM')[cols])
rank_dfs = {}

for i in [1.0, 0.75, 0.5, 0.25, 0.0]:
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    ratio_label = f"Ratio_SBM_{sbm_val}"
    G_name = f"artificial_graph_sbmv5_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    G_train, dataset_train, dataset_eval, xgboost_data, _, _ = gp.load_all_data_for_graph(G_name)
    model = xgboost_data['model']
    importance_gain = model.get_booster().get_score(importance_type='gain')
    
    df = pd.DataFrame(list(importance_gain.items()), columns=['Feature', 'Gain'])
    df = df.sort_values(by='Gain', ascending=False).reset_index(drop=True)
    
    df['Rank'] = df.index + 1
    df['Weight_pct'] = (df['Gain'] / df['Gain'].sum()) * 100
    
    rank_dfs[ratio_label] = df

# --- 1. PRÉPARATION DES DONNÉES ---
if "Ratio_SBM_1.00" in rank_dfs and "Ratio_SBM_0.00" in rank_dfs:
    df_100 = rank_dfs["Ratio_SBM_1.00"]
    df_000 = rank_dfs["Ratio_SBM_0.00"]
    
    # Merge
    comparison = pd.merge(
        df_100, df_000, on='Feature', 
        suffixes='_SBM', '_Spatial', how='outer')
    
    # --- 2. TRAITEMENT DES FEATURES ABSENTES (Anti-NaN) ---
    # Si une feature est absente, son poids est 0 et son rang est le max (ex: 100)
    comparison['W_pct_SBM'] = comparison['W_pct_SBM'].fillna(0)
    comparison['W_pct_Spatial'] = comparison['W_pct_Spatial'].fillna(0)
    comparison['Gain_Raw_SBM'] = comparison['Gain_Raw_SBM'].fillna(0)
    comparison['Gain_Raw_Spatial'] = comparison['Gain_Raw_Spatial'].fillna(0)
    
    # Pour le rang, on met une valeur arbitraire haute (ex: 100) pour marquer l'absence
    comparison['Rank_SBM'] = comparison['Rank_SBM'].fillna(100)
    comparison['Rank_Spatial'] = comparison['Rank_Spatial'].fillna(100)
    
    # --- 3. CALCUL DES MÉTRIQUES D'ANALYSE ---
    comparison['Rank_Shift'] = comparison['Rank_Spatial'] - comparison['Rank_SBM']
    
    # On calcule la différence de poids relatif (très parlant pour l'explainability)
    comparison['Weight_Delta'] = comparison['W_pct_SBM'] - comparison['W_pct_Spatial']

    # --- 4. FILTRAGE DES "VRAIS" ACTEURS ---
    # On ne s'intéresse qu'aux features qui "existent" significativement (Top 15 d'un côté ou de l'autre)
    top_features_mask = (comparison['Rank_SBM'] <= 15) | (comparison['Rank_Spatial'] <= 15)
    relevant_comparison = comparison[top_features_mask].copy()

    # Affichage
    pd.options.display.float_format = '{:.2f}'.format
    cols = ['Feature', 'Rank_SBM', 'W_pct_SBM', 'Rank_Spatial', 'W_pct_Spatial', 'Rank_Shift', 'Weight_Delta']

    print("\n" + "="*90)
    print("🎯 ANALYSE DES ACTEURS MAJEURS (TOP 15 SBM OU SPATIAL)")
    print("="*90)

    print("\n🔥 MARQUEURS SBM (Dominent le SBM, chutent en Spatial) :")
    # Tri par Weight_Delta positif (poids SBM >> poids Spatial)
    print(relevant_comparison.sort_values('Weight_Delta', ascending=False).head(10)[cols])

    print("\n🌍 MARQUEURS SPATIAUX (Dominent le Spatial, chutent en SBM) :")
    # Tri par Weight_Delta négatif (poids Spatial >> poids SBM)
    print(relevant_comparison.sort_values('Weight_Delta', ascending=True).head(10)[cols])

    print("\n🧱 PILIERS STRUCTURELS (Importants partout) :")
    # Features dans le top 15 des deux côtés avec un Delta faible
    piliers = relevant_comparison[
        (relevant_comparison['Rank_SBM'] <= 15) & 
        (relevant_comparison['Rank_Spatial'] <= 15) &
        (relevant_comparison['Weight_Delta'].abs() < 5)
    ]
    print(piliers.sort_values('Rank_SBM')[cols])
rank_dfs = {}

for i in [1.0, 0.75, 0.5, 0.25, 0.0]:
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    ratio_label = f"Ratio_SBM_{sbm_val}"
    G_name = f"artificial_graph_sbmv5_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    G_train, dataset_train, dataset_eval, xgboost_data, _, _ = gp.load_all_data_for_graph(G_name)
    model = xgboost_data['model']
    importance_gain = model.get_booster().get_score(importance_type='gain')
    
    df = pd.DataFrame(list(importance_gain.items()), columns=['Feature', 'Gain'])
    df = df.sort_values(by='Gain', ascending=False).reset_index(drop=True)
    
    df['Rank'] = df.index + 1
    df['Weight_pct'] = (df['Gain'] / df['Gain'].sum()) * 100
    
    rank_dfs[ratio_label] = df

# --- 1. PRÉPARATION DES DONNÉES ---
if "Ratio_SBM_1.00" in rank_dfs and "Ratio_SBM_0.00" in rank_dfs:
    df_100 = rank_dfs["Ratio_SBM_1.00"]
    df_000 = rank_dfs["Ratio_SBM_0.00"]
    
    # Merge
    comparison = pd.merge(df_100, df_000, on='Feature', suffixes='_SBM', '_Spatial', how='outer')
    
    # --- 2. TRAITEMENT DES FEATURES ABSENTES (Anti-NaN) ---
    # Si une feature est absente, son poids est 0 et son rang est le max (ex: 100)
    comparison['W_pct_SBM'] = comparison['W_pct_SBM'].fillna(0)
    comparison['W_pct_Spatial'] = comparison['W_pct_Spatial'].fillna(0)
    comparison['Gain_Raw_SBM'] = comparison['Gain_Raw_SBM'].fillna(0)
    comparison['Gain_Raw_Spatial'] = comparison['Gain_Raw_Spatial'].fillna(0)
    
    # Pour le rang, on met une valeur arbitraire haute (ex: 100) pour marquer l'absence
    comparison['Rank_SBM'] = comparison['Rank_SBM'].fillna(100)
    comparison['Rank_Spatial'] = comparison['Rank_Spatial'].fillna(100)
    
    # --- 3. CALCUL DES MÉTRIQUES D'ANALYSE ---
    comparison['Rank_Shift'] = comparison['Rank_Spatial'] - comparison['Rank_SBM']
    
    # On calcule la différence de poids relatif (très parlant pour l'explainability)
    comparison['Weight_Delta'] = comparison['W_pct_SBM'] - comparison['W_pct_Spatial']

    # --- 4. FILTRAGE DES "VRAIS" ACTEURS ---
    # On ne s'intéresse qu'aux features qui "existent" significativement (Top 15 d'un côté ou de l'autre)
    top_features_mask = (comparison['Rank_SBM'] <= 15) | (comparison['Rank_Spatial'] <= 15)
    relevant_comparison = comparison[top_features_mask].copy()

    # Affichage
    pd.options.display.float_format = '{:.2f}'.format
    cols = ['Feature', 'Rank_SBM', 'W_pct_SBM', 'Rank_Spatial', 'W_pct_Spatial', 'Rank_Shift', 'Weight_Delta']

    print("\n" + "="*90)
    print("🎯 ANALYSE DES ACTEURS MAJEURS (TOP 15 SBM OU SPATIAL)")
    print("="*90)

    print("\n🔥 MARQUEURS SBM (Dominent le SBM, chutent en Spatial) :")
    # Tri par Weight_Delta positif (poids SBM >> poids Spatial)
    print(relevant_comparison.sort_values('Weight_Delta', ascending=False).head(10)[cols])

    print("\n🌍 MARQUEURS SPATIAUX (Dominent le Spatial, chutent en SBM) :")
    # Tri par Weight_Delta négatif (poids Spatial >> poids SBM)
    print(relevant_comparison.sort_values('Weight_Delta', ascending=True).head(10)[cols])

    print("\n🧱 PILIERS STRUCTURELS (Importants partout) :")
    # Features dans le top 15 des deux côtés avec un Delta faible
    piliers = relevant_comparison[
        (relevant_comparison['Rank_SBM'] <= 15) & 
        (relevant_comparison['Rank_Spatial'] <= 15) &
        (relevant_comparison['Weight_Delta'].abs() < 5)
    ]
    print(piliers.sort_values('Rank_SBM')[cols])
rank_dfs = {}

for i in [1.0, 0.75, 0.5, 0.25, 0.0]:
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    ratio_label = f"Ratio_SBM_{sbm_val}"
    G_name = f"artificial_graph_sbmv5_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    G_train, dataset_train, dataset_eval, xgboost_data, _, _ = gp.load_all_data_for_graph(G_name)
    model = xgboost_data['model']
    importance_gain = model.get_booster().get_score(importance_type='gain')
    
    df = pd.DataFrame(list(importance_gain.items()), columns=['Feature', 'Gain'])
    df = df.sort_values(by='Gain', ascending=False).reset_index(drop=True)
    
    df['Rank'] = df.index + 1
    df['Weight_pct'] = (df['Gain'] / df['Gain'].sum()) * 100
    
    rank_dfs[ratio_label] = df

import pandas as pd
import numpy as np

# --- 1. PRÉPARATION DES DONNÉES ---
# Assure-toi que rank_dfs["Ratio_SBM_1.00"] et rank_dfs["Ratio_SBM_0.00"] existent

if "Ratio_SBM_1.00" in rank_dfs and "Ratio_SBM_0.00" in rank_dfs:
    df_100 = rank_dfs["Ratio_SBM_1.00"]
    df_000 = rank_dfs["Ratio_SBM_0.00"]
    
    # --- CORRECTION ICI : suffixes=('SBM', 'Spatial') dans un tuple ---
    comparison = pd.merge(
        df_100, 
        df_000, 
        on='Feature', 
        suffixes=('_SBM', '_Spatial'), 
        how='outer'
    )
    
    # --- 2. TRAITEMENT DES FEATURES ABSENTES ---
    comparison['W_pct_SBM'] = comparison['W_pct_SBM'].fillna(0.0)
    comparison['W_pct_Spatial'] = comparison['W_pct_Spatial'].fillna(0.0)
    comparison['Gain_Raw_SBM'] = comparison['Gain_Raw_SBM'].fillna(0.0)
    comparison['Gain_Raw_Spatial'] = comparison['Gain_Raw_Spatial'].fillna(0.0)
    
    # Pour le rang, on met 99 pour marquer le fond de classement
    comparison['Rank_SBM'] = comparison['Rank_SBM'].fillna(99)
    comparison['Rank_Spatial'] = comparison['Rank_Spatial'].fillna(99)
    
    # --- 3. CALCUL DES MÉTRIQUES ---
    comparison['Rank_Shift'] = comparison['Rank_Spatial'] - comparison['Rank_SBM']
    comparison['Weight_Delta'] = comparison['W_pct_SBM'] - comparison['W_pct_Spatial']

    # --- 4. FILTRAGE DES ACTEURS MAJEURS ---
    # On garde ce qui est dans le Top 15 (SBM ou Spatial) OU ce qui a un poids > 1%
    top_mask = (comparison['Rank_SBM'] <= 15) | (comparison['Rank_Spatial'] <= 15)
    relevant_comparison = comparison[top_mask].copy()

    # Affichage propre
    pd.options.display.float_format = '{:.2f}'.format
    cols = ['Feature', 'Rank_SBM', 'W_pct_SBM', 'Rank_Spatial', 'W_pct_Spatial', 'Rank_Shift', 'Weight_Delta']

    print("\n" + "="*95)
    print("🎯 ANALYSE DES ACTEURS MAJEURS (TOP 15 SBM OU SPATIAL)")
    print("="*95)

    print("\n🔥 MARQUEURS SBM (Poids SBM > Poids Spatial) :")
    print(relevant_comparison.sort_values('Weight_Delta', ascending=False).head(10)[cols])

    print("\n🌍 MARQUEURS SPATIAUX (Poids Spatial > Poids SBM) :")
    print(relevant_comparison.sort_values('Weight_Delta', ascending=True).head(10)[cols])

    print("\n🧱 PILIERS STRUCTURELS (Top 15 stables, Delta < 5%) :")
    piliers = relevant_comparison[
        (relevant_comparison['Rank_SBM'] <= 15) & 
        (relevant_comparison['Rank_Spatial'] <= 15) &
        (relevant_comparison['Weight_Delta'].abs() < 5)
    ]
    print(piliers.sort_values('Rank_SBM')[cols])
# --- 1. GÉNÉRATION DES DICTIONNAIRES D'IMPORTANCE ---
rank_dfs = {}

for i in [1.0, 0.75, 0.5, 0.25, 0.0]:
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    ratio_label = f"Ratio_SBM_{sbm_val}"
    G_name = f"artificial_graph_sbmv5_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    try:
        # Chargement
        _, _, _, xgboost_data, _, _ = gp.load_all_data_for_graph(G_name)
        model = xgboost_data['model']
        
        # Gain du booster
        importance_gain = model.get_booster().get_score(importance_type='gain')
        
        # Création du DF
        df = pd.DataFrame(list(importance_gain.items()), columns=['Feature', 'Gain'])
        df = df.sort_values(by='Gain', ascending=False).reset_index(drop=True)
        
        # Calcul Rang et Poids
        df['Rank'] = df.index + 1
        df['Weight_pct'] = (df['Gain'] / df['Gain'].sum()) * 100
        
        rank_dfs[ratio_label] = df
        print(f"✅ {ratio_label} chargé.")
    except Exception as e:
        print(f"⚠️ Erreur sur {G_name}: {e}")

# --- 2. ANALYSE ET COMPARAISON ---

if "Ratio_SBM_1.00" in rank_dfs and "Ratio_SBM_0.00" in rank_dfs:
    # On isole les deux extrêmes
    df_100 = rank_dfs["Ratio_SBM_1.00"].copy()
    df_000 = rank_dfs["Ratio_SBM_0.00"].copy()
    
    # Merge avec suffixes explicites
    comparison = pd.merge(
        df_100, 
        df_000, 
        on='Feature', 
        suffixes=('_SBM', '_Spatial'), 
        how='outer'
    )
    
    # --- 3. NETTOYAGE DES ABSENTS (Anti-KeyError / Anti-NaN) ---
    # On utilise les noms exacts générés par le merge : 'Weight_pct_SBM', etc.
    cols_to_zero = ['Weight_pct_SBM', 'Weight_pct_Spatial', 'Gain_SBM', 'Gain_Spatial']
    for col in cols_to_zero:
        if col in comparison.columns:
            comparison[col] = comparison[col].fillna(0.0)
    
    comparison['Rank_SBM'] = comparison['Rank_SBM'].fillna(99)
    comparison['Rank_Spatial'] = comparison['Rank_Spatial'].fillna(99)
    
    # --- 4. CALCUL DES MÉTRIQUES ---
    comparison['Rank_Shift'] = comparison['Rank_Spatial'] - comparison['Rank_SBM']
    comparison['Weight_Delta'] = comparison['Weight_pct_SBM'] - comparison['Weight_pct_Spatial']

    # --- 5. FILTRAGE ET AFFICHAGE ---
    # On garde le Top 15 de l'un ou l'autre
    top_mask = (comparison['Rank_SBM'] <= 15) | (comparison['Rank_Spatial'] <= 15)
    relevant = comparison[top_mask].copy()

    pd.options.display.float_format = '{:.2f}'.format
    # On ajuste la liste des colonnes pour l'affichage final
    display_cols = ['Feature', 'Rank_SBM', 'Weight_pct_SBM', 'Rank_Spatial', 'Weight_pct_Spatial', 'Rank_Shift', 'Weight_Delta']

    print("\n" + "="*100)
    print("🎯 ANALYSE DES ACTEURS MAJEURS")
    print("="*100)

    print("\n🔥 MARQUEURS SBM (Spécialistes Communauté) :")
    print(relevant.sort_values('Weight_Delta', ascending=False).head(10)[display_cols])

    print("\n🌍 MARQUEURS SPATIAUX (Spécialistes Géométrie) :")
    print(relevant.sort_values('Weight_Delta', ascending=True).head(10)[display_cols])

    print("\n🧱 PILIERS STRUCTURELS (Polyvalents) :")
    piliers = relevant[
        (relevant['Rank_SBM'] <= 15) & 
        (relevant['Rank_Spatial'] <= 15) &
        (relevant['Weight_Delta'].abs() < 5)
    ]
    print(piliers.sort_values('Rank_SBM')[display_cols])
# --- 1. GÉNÉRATION DES DICTIONNAIRES D'IMPORTANCE ---
rank_dfs = {}

for i in [1.0, 0.75, 0.5, 0.25, 0.0]:
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    ratio_label = f"Ratio_SBM_{sbm_val}"
    G_name = f"artificial_graph_sbmv5_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    _, _, _, xgboost_data, _, _ = gp.load_all_data_for_graph(G_name)
    model = xgboost_data['model']
    
    # Gain du booster
    importance_gain = model.get_booster().get_score(importance_type='gain')
    
    # Création du DF
    df = pd.DataFrame(list(importance_gain.items()), columns=['Feature', 'Gain'])
    df = df.sort_values(by='Gain', ascending=False).reset_index(drop=True)
    
    # Calcul Rang et Poids
    df['Rank'] = df.index + 1
    df['Weight_pct'] = (df['Gain'] / df['Gain'].sum()) * 100
    
    rank_dfs[ratio_label] = df
    print(f"{ratio_label} chargé.")


if "Ratio_SBM_1.00" in rank_dfs and "Ratio_SBM_0.00" in rank_dfs:
    # 1. Préparation des données sources
    df_sbm = rank_dfs["Ratio_SBM_1.00"].copy()
    df_spa = rank_dfs["Ratio_SBM_0.00"].copy()
    
    # 2. Merge avec renommage propre
    comparison = pd.merge(
        df_sbm[['Feature', 'Rank', 'Weight_pct']], 
        df_spa[['Feature', 'Rank', 'Weight_pct']], 
        on='Feature', 
        suffixes=('_SBM', '_Spatial'), 
        how='outer'
    )
    
    # 3. Nettoyage des valeurs manquantes
    comparison['Weight_pct_SBM'] = comparison['Weight_pct_SBM'].fillna(0.0)
    comparison['Weight_pct_Spatial'] = comparison['Weight_pct_Spatial'].fillna(0.0)
    comparison['Rank_SBM'] = comparison['Rank_SBM'].fillna(99).astype(int)
    comparison['Rank_Spatial'] = comparison['Rank_Spatial'].fillna(99).astype(int)
    
    # 4. Calcul des colonnes de tri et d'analyse
    comparison['Sum_Weight'] = comparison['Weight_pct_SBM'] + comparison['Weight_pct_Spatial']
    comparison['Weight_Delta'] = comparison['Weight_pct_SBM'] - comparison['Weight_pct_Spatial']
    
    # 5. Filtrage : au moins 0.5% dans l'un des deux mondes
    mask = (comparison['Weight_pct_SBM'] >= 0.5) | (comparison['Weight_pct_Spatial'] >= 0.5)
    final_table = comparison[mask].copy()
    
    # 6. Tri par importance cumulée
    final_table = final_table.sort_values(by='Sum_Weight', ascending=False)
    
    # Affichage final
    pd.options.display.float_format = '{:.2f}'.format
    cols_to_print = [
        'Feature', 
        'Weight_pct_SBM', 'Rank_SBM', 
        'Weight_pct_Spatial', 'Rank_Spatial', 
        'Weight_Delta', 'Sum_Weight'
    ]
    
    print("Tableau des Features Significatives (Seuil > 0.5%)")
    print("-" * 100)
    print(final_table[cols_to_print].to_string(index=False))
# --- 1. GÉNÉRATION DES DICTIONNAIRES D'IMPORTANCE ---
rank_dfs = {}

for i in [1.0, 0.75, 0.5, 0.25, 0.0]:
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    ratio_label = f"Ratio_SBM_{sbm_val}"
    G_name = f"artificial_graph_sbmv4_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    _, _, _, xgboost_data, _, _ = gp.load_all_data_for_graph(G_name)
    model = xgboost_data['model']
    
    # Gain du booster
    importance_gain = model.get_booster().get_score(importance_type='gain')
    
    # Création du DF
    df = pd.DataFrame(list(importance_gain.items()), columns=['Feature', 'Gain'])
    df = df.sort_values(by='Gain', ascending=False).reset_index(drop=True)
    
    # Calcul Rang et Poids
    df['Rank'] = df.index + 1
    df['Weight_pct'] = (df['Gain'] / df['Gain'].sum()) * 100
    
    rank_dfs[ratio_label] = df
    print(f"{ratio_label} chargé.")


if "Ratio_SBM_1.00" in rank_dfs and "Ratio_SBM_0.00" in rank_dfs:
    # 1. Préparation des données sources
    df_sbm = rank_dfs["Ratio_SBM_1.00"].copy()
    df_spa = rank_dfs["Ratio_SBM_0.00"].copy()
    
    # 2. Merge avec renommage propre
    comparison = pd.merge(
        df_sbm[['Feature', 'Rank', 'Weight_pct']], 
        df_spa[['Feature', 'Rank', 'Weight_pct']], 
        on='Feature', 
        suffixes=('_SBM', '_Spatial'), 
        how='outer'
    )
    
    # 3. Nettoyage des valeurs manquantes
    comparison['Weight_pct_SBM'] = comparison['Weight_pct_SBM'].fillna(0.0)
    comparison['Weight_pct_Spatial'] = comparison['Weight_pct_Spatial'].fillna(0.0)
    comparison['Rank_SBM'] = comparison['Rank_SBM'].fillna(99).astype(int)
    comparison['Rank_Spatial'] = comparison['Rank_Spatial'].fillna(99).astype(int)
    
    # 4. Calcul des colonnes de tri et d'analyse
    comparison['Sum_Weight'] = comparison['Weight_pct_SBM'] + comparison['Weight_pct_Spatial']
    comparison['Weight_Delta'] = comparison['Weight_pct_SBM'] - comparison['Weight_pct_Spatial']
    
    # 5. Filtrage : au moins 0.5% dans l'un des deux mondes
    mask = (comparison['Weight_pct_SBM'] >= 0.5) | (comparison['Weight_pct_Spatial'] >= 0.5)
    final_table = comparison[mask].copy()
    
    # 6. Tri par importance cumulée
    final_table = final_table.sort_values(by='Sum_Weight', ascending=False)
    
    # Affichage final
    pd.options.display.float_format = '{:.2f}'.format
    cols_to_print = [
        'Feature', 
        'Weight_pct_SBM', 'Rank_SBM', 
        'Weight_pct_Spatial', 'Rank_Spatial', 
        'Weight_Delta', 'Sum_Weight'
    ]
    
    print("Tableau des Features Significatives (Seuil > 0.5%)")
    print("-" * 100)
    print(final_table[cols_to_print].to_string(index=False))
# --- 1. GÉNÉRATION DES DICTIONNAIRES D'IMPORTANCE ---
rank_dfs = {}

for i in [1.0, 0.75, 0.5, 0.25, 0.0]:
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    ratio_label = f"Ratio_SBM_{sbm_val}"
    G_name = f"artificial_graph_sbmv5_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    _, _, _, xgboost_data, _, _ = gp.load_all_data_for_graph(G_name)
    model = xgboost_data['model']
    
    # Gain du booster
    importance_gain = model.get_booster().get_score(importance_type='gain')
    
    # Création du DF
    df = pd.DataFrame(list(importance_gain.items()), columns=['Feature', 'Gain'])
    df = df.sort_values(by='Gain', ascending=False).reset_index(drop=True)
    
    # Calcul Rang et Poids
    df['Rank'] = df.index + 1
    df['Weight_pct'] = (df['Gain'] / df['Gain'].sum()) * 100
    
    rank_dfs[ratio_label] = df
    print(f"{ratio_label} chargé.")


if "Ratio_SBM_1.00" in rank_dfs and "Ratio_SBM_0.00" in rank_dfs:
    # 1. Préparation des données sources
    df_sbm = rank_dfs["Ratio_SBM_1.00"].copy()
    df_spa = rank_dfs["Ratio_SBM_0.00"].copy()
    
    # 2. Merge avec renommage propre
    comparison = pd.merge(
        df_sbm[['Feature', 'Rank', 'Weight_pct']], 
        df_spa[['Feature', 'Rank', 'Weight_pct']], 
        on='Feature', 
        suffixes=('_SBM', '_Spatial'), 
        how='outer'
    )
    
    # 3. Nettoyage des valeurs manquantes
    comparison['Weight_pct_SBM'] = comparison['Weight_pct_SBM'].fillna(0.0)
    comparison['Weight_pct_Spatial'] = comparison['Weight_pct_Spatial'].fillna(0.0)
    comparison['Rank_SBM'] = comparison['Rank_SBM'].fillna(99).astype(int)
    comparison['Rank_Spatial'] = comparison['Rank_Spatial'].fillna(99).astype(int)
    
    # 4. Calcul des colonnes de tri et d'analyse
    comparison['Sum_Weight'] = comparison['Weight_pct_SBM'] + comparison['Weight_pct_Spatial']
    comparison['Weight_Delta'] = comparison['Weight_pct_SBM'] - comparison['Weight_pct_Spatial']
    
    # 5. Filtrage : au moins 0.5% dans l'un des deux mondes
    mask = (comparison['Weight_pct_SBM'] >= 0.5) | (comparison['Weight_pct_Spatial'] >= 0.5)
    final_table = comparison[mask].copy()
    
    # 6. Tri par importance cumulée
    final_table = final_table.sort_values(by='Sum_Weight', ascending=False)
    
    # Affichage final
    pd.options.display.float_format = '{:.2f}'.format
    cols_to_print = [
        'Feature', 
        'Weight_pct_SBM', 'Rank_SBM', 
        'Weight_pct_Spatial', 'Rank_Spatial', 
        'Weight_Delta', 'Sum_Weight'
    ]
    
    print("Tableau des Features Significatives (Seuil > 0.5%)")
    print("-" * 100)
    print(final_table[cols_to_print].to_string(index=False))
# --- 1. GÉNÉRATION DES DICTIONNAIRES D'IMPORTANCE ---
rank_dfs = {}

for i in [1.0, 0.75, 0.5, 0.25, 0.0]:
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    ratio_label = f"Ratio_SBM_{sbm_val}"
    G_name = f"artificial_graph_sbmv4_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    _, _, _, xgboost_data, _, _ = gp.load_all_data_for_graph(G_name)
    model = xgboost_data['model']
    
    # Gain du booster
    importance_gain = model.get_booster().get_score(importance_type='gain')
    
    # Création du DF
    df = pd.DataFrame(list(importance_gain.items()), columns=['Feature', 'Gain'])
    df = df.sort_values(by='Gain', ascending=False).reset_index(drop=True)
    
    # Calcul Rang et Poids
    df['Rank'] = df.index + 1
    df['Weight_pct'] = (df['Gain'] / df['Gain'].sum()) * 100
    
    rank_dfs[ratio_label] = df
    print(f"{ratio_label} chargé.")


if "Ratio_SBM_1.00" in rank_dfs and "Ratio_SBM_0.00" in rank_dfs:
    # 1. Préparation des données sources
    df_sbm = rank_dfs["Ratio_SBM_1.00"].copy()
    df_spa = rank_dfs["Ratio_SBM_0.00"].copy()
    
    # 2. Merge avec renommage propre
    comparison = pd.merge(
        df_sbm[['Feature', 'Rank', 'Weight_pct']], 
        df_spa[['Feature', 'Rank', 'Weight_pct']], 
        on='Feature', 
        suffixes=('_SBM', '_Spatial'), 
        how='outer'
    )
    
    # 3. Nettoyage des valeurs manquantes
    comparison['Weight_pct_SBM'] = comparison['Weight_pct_SBM'].fillna(0.0)
    comparison['Weight_pct_Spatial'] = comparison['Weight_pct_Spatial'].fillna(0.0)
    comparison['Rank_SBM'] = comparison['Rank_SBM'].fillna(99).astype(int)
    comparison['Rank_Spatial'] = comparison['Rank_Spatial'].fillna(99).astype(int)
    
    # 4. Calcul des colonnes de tri et d'analyse
    comparison['Sum_Weight'] = comparison['Weight_pct_SBM'] + comparison['Weight_pct_Spatial']
    comparison['Weight_Delta'] = comparison['Weight_pct_SBM'] - comparison['Weight_pct_Spatial']
    
    # 5. Filtrage : au moins 0.5% dans l'un des deux mondes
    mask = (comparison['Weight_pct_SBM'] >= 0.5) | (comparison['Weight_pct_Spatial'] >= 0.5)
    final_table = comparison[mask].copy()
    
    # 6. Tri par importance cumulée
    final_table = final_table.sort_values(by='Sum_Weight', ascending=False)
    
    # Affichage final
    pd.options.display.float_format = '{:.2f}'.format
    cols_to_print = [
        'Feature', 
        'Weight_pct_SBM', 'Rank_SBM', 
        'Weight_pct_Spatial', 'Rank_Spatial', 
        'Weight_Delta', 'Sum_Weight'
    ]
    
    print("Tableau des Features Significatives (Seuil > 0.5%)")
    print("-" * 100)
    print(final_table[cols_to_print].to_string(index=False))
all_summaries = []

for i in np.arange(0, 1.25, 0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv4_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    _, _, _, xgboost_data, shap_explainer, _ = gp.load_all_data_for_graph(G_name)
    X_test = xgboost_data['X_test']

    print(f"Type de l'objet : {type(shap_explainer)}")
    print(f"Dimensions des valeurs : {shap_explainer.values.shape}")

    vals = shap_explainer.values
    
    if isinstance(vals, list):
        vals = vals[1]

    df_shap = pd.DataFrame(vals, columns=X_test.columns)
    mean_shap = df_shap.mean()
    mean_abs_shap = df_shap.abs().mean()
    
    summary = pd.DataFrame({
        'feature': X_test.columns,
        'mean_shap': mean_shap.values,
        'mean_abs_shap': mean_abs_shap.values,
        'sbm_param': i
    })
    all_summaries.append(summary)

df_shapvals_concat = pd.concat(all_summaries)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_importance_evolution(df_results, metric="mean_shap", as_ratio=True):
    """
    Génère une heatmap montrant l'évolution de l'importance des features (SHAP).
    """
    heatmap_data = df_results.pivot(index='feature', columns='sbm_param', values=metric)
    
    if as_ratio:
        heatmap_data = heatmap_data.div(heatmap_data.sum(axis=0), axis=1)
        label_name = f"Ratio de contribution ({metric})"
        fmt_type = ".2%"
    else:
        label_name = f"Valeur brute ({metric})"
        fmt_type = ".3f"

    heatmap_data['avg_importance'] = heatmap_data.mean(axis=1)
    heatmap_data = heatmap_data.sort_values(by='avg_importance', ascending=False).drop(columns='avg_importance')

    plt.figure(figsize=(16, 9))
    sns.heatmap(heatmap_data, annot=True, fmt=fmt_type, cmap="YlGnBu", yticklabels=True, cbar_kws={'label': label_name})

    title_suffix = "(Ratios normalisés)" if as_ratio else "(Valeurs brutes)"
    plt.title(f"Importance relative des features (SHAP) selon la structure du graphe {title_suffix}")
    plt.xlabel("Paramètre du graphe (i)")
    plt.ylabel("Features")
    plt.tight_layout()

    filename = f"outputs/plots/heatmap_feat_importance_per_ratio_v4_{metric}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé sous : {filename}")
    plt.show()

plot_feature_importance_evolution(df_shapvals_concat, metric="mean_abs_shap", as_ratio=False)
print(df_shapvals_concat.columns)
features_list = df_shapvals_concat['feature']
print(", ".join(map(str, features_list)))

"""
all_distribs = []
for i in np.arange(0, 1.05, 0.05):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbm_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    _, _,_, xgboost_data, _, _ = gp.load_all_data_for_graph(G_name)
    X_test = xgboost_data['X_test']
    y_test = xgboost_data['y_test']
    
    pos_sp = X_test[y_test == 1]['sp'].value_counts(normalize=True).to_dict()
    neg_sp = X_test[y_test == 0]['sp'].value_counts(normalize=True).to_dict()
    
    row = {'sbm_param': round(i, 2)}
    for dist in range(1, 7):
        row[f'pos_sp_{dist}'] = pos_sp.get(dist, 0)
        row[f'neg_sp_{dist}'] = neg_sp.get(dist, 0)
    
    row['pos_sp_7+'] = sum(v for k, v in pos_sp.items() if k >= 7 or k == 0)
    row['neg_sp_7+'] = sum(v for k, v in neg_sp.items() if k >= 7 or k == 0)
    
    all_distribs.append(row)

df_sp_detailed = pd.DataFrame(all_distribs)

cols = ['sbm_param'] + [f'{prefix}_sp_{d}' for d in list(range(1, 7)) + ['7+'] for prefix in ['pos', 'neg']]
df_sp_detailed = df_sp_detailed[cols]

pd.options.display.float_format = '{:.3f}'.format
print(df_sp_detailed.to_string(index=False))
"""
all_summaries = []

for i in np.arange(0, 1.25, 0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv5_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    _, _, _, xgboost_data, shap_explainer, _ = gp.load_all_data_for_graph(G_name)
    X_test = xgboost_data['X_test']

    print(f"Type de l'objet : {type(shap_explainer)}")
    print(f"Dimensions des valeurs : {shap_explainer.values.shape}")

    vals = shap_explainer.values
    
    if isinstance(vals, list):
        vals = vals[1]

    df_shap = pd.DataFrame(vals, columns=X_test.columns)
    mean_shap = df_shap.mean()
    mean_abs_shap = df_shap.abs().mean()
    
    summary = pd.DataFrame({
        'feature': X_test.columns,
        'mean_shap': mean_shap.values,
        'mean_abs_shap': mean_abs_shap.values,
        'sbm_param': i
    })
    all_summaries.append(summary)

df_shapvals_concat = pd.concat(all_summaries)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_importance_evolution(df_results, metric="mean_shap", as_ratio=True):
    """
    Génère une heatmap montrant l'évolution de l'importance des features (SHAP).
    """
    heatmap_data = df_results.pivot(index='feature', columns='sbm_param', values=metric)
    
    if as_ratio:
        heatmap_data = heatmap_data.div(heatmap_data.sum(axis=0), axis=1)
        label_name = f"Ratio de contribution ({metric})"
        fmt_type = ".2%"
    else:
        label_name = f"Valeur brute ({metric})"
        fmt_type = ".3f"

    heatmap_data['avg_importance'] = heatmap_data.mean(axis=1)
    heatmap_data = heatmap_data.sort_values(by='avg_importance', ascending=False).drop(columns='avg_importance')

    plt.figure(figsize=(16, 9))
    sns.heatmap(heatmap_data, annot=True, fmt=fmt_type, cmap="YlGnBu", yticklabels=True, cbar_kws={'label': label_name})

    title_suffix = "(Ratios normalisés)" if as_ratio else "(Valeurs brutes)"
    plt.title(f"Importance relative des features (SHAP) selon la structure du graphe {title_suffix}")
    plt.xlabel("Paramètre du graphe (i)")
    plt.ylabel("Features")
    plt.tight_layout()

    filename = f"outputs/plots/heatmap_feat_importance_per_ratio_v4_{metric}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé sous : {filename}")
    plt.show()

plot_feature_importance_evolution(df_shapvals_concat, metric="mean_abs_shap", as_ratio=False)
print(df_shapvals_concat.columns)
features_list = df_shapvals_concat['feature']
print(", ".join(map(str, features_list)))

"""
all_distribs = []
for i in np.arange(0, 1.05, 0.05):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbm_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    _, _,_, xgboost_data, _, _ = gp.load_all_data_for_graph(G_name)
    X_test = xgboost_data['X_test']
    y_test = xgboost_data['y_test']
    
    pos_sp = X_test[y_test == 1]['sp'].value_counts(normalize=True).to_dict()
    neg_sp = X_test[y_test == 0]['sp'].value_counts(normalize=True).to_dict()
    
    row = {'sbm_param': round(i, 2)}
    for dist in range(1, 7):
        row[f'pos_sp_{dist}'] = pos_sp.get(dist, 0)
        row[f'neg_sp_{dist}'] = neg_sp.get(dist, 0)
    
    row['pos_sp_7+'] = sum(v for k, v in pos_sp.items() if k >= 7 or k == 0)
    row['neg_sp_7+'] = sum(v for k, v in neg_sp.items() if k >= 7 or k == 0)
    
    all_distribs.append(row)

df_sp_detailed = pd.DataFrame(all_distribs)

cols = ['sbm_param'] + [f'{prefix}_sp_{d}' for d in list(range(1, 7)) + ['7+'] for prefix in ['pos', 'neg']]
df_sp_detailed = df_sp_detailed[cols]

pd.options.display.float_format = '{:.3f}'.format
print(df_sp_detailed.to_string(index=False))
"""
%load_ext autoreload
%autoreload 2

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
all_summaries = []

for i in np.arange(0, 1.25, 0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv5_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    _, _, _, xgboost_data, shap_explainer, _ = gp.load_all_data_for_graph(G_name)
    X_test = xgboost_data['X_test']

    print(f"Type de l'objet : {type(shap_explainer)}")
    print(f"Dimensions des valeurs : {shap_explainer.values.shape}")

    vals = shap_explainer.values
    
    if isinstance(vals, list):
        vals = vals[1]

    df_shap = pd.DataFrame(vals, columns=X_test.columns)
    mean_shap = df_shap.mean()
    mean_abs_shap = df_shap.abs().mean()
    
    summary = pd.DataFrame({
        'feature': X_test.columns,
        'mean_shap': mean_shap.values,
        'mean_abs_shap': mean_abs_shap.values,
        'sbm_param': i
    })
    all_summaries.append(summary)

df_shapvals_concat = pd.concat(all_summaries)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_importance_evolution(df_results, metric="mean_shap", as_ratio=True):
    """
    Génère une heatmap montrant l'évolution de l'importance des features (SHAP).
    """
    heatmap_data = df_results.pivot(index='feature', columns='sbm_param', values=metric)
    
    if as_ratio:
        heatmap_data = heatmap_data.div(heatmap_data.sum(axis=0), axis=1)
        label_name = f"Ratio de contribution ({metric})"
        fmt_type = ".2%"
    else:
        label_name = f"Valeur brute ({metric})"
        fmt_type = ".3f"

    heatmap_data['avg_importance'] = heatmap_data.mean(axis=1)
    heatmap_data = heatmap_data.sort_values(by='avg_importance', ascending=False).drop(columns='avg_importance')

    plt.figure(figsize=(16, 9))
    sns.heatmap(heatmap_data, annot=True, fmt=fmt_type, cmap="YlGnBu", yticklabels=True, cbar_kws={'label': label_name})

    title_suffix = "(Ratios normalisés)" if as_ratio else "(Valeurs brutes)"
    plt.title(f"Importance relative des features (SHAP) selon la structure du graphe {title_suffix}")
    plt.xlabel("Paramètre du graphe (i)")
    plt.ylabel("Features")
    plt.tight_layout()

    filename = f"outputs/plots/heatmap_feat_importance_per_ratio_v4_{metric}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé sous : {filename}")
    plt.show()

plot_feature_importance_evolution(df_shapvals_concat, metric="mean_abs_shap", as_ratio=False)
print(df_shapvals_concat.columns)
features_list = df_shapvals_concat['feature']
print(", ".join(map(str, features_list)))

"""
all_distribs = []
for i in np.arange(0, 1.05, 0.05):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbm_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    _, _,_, xgboost_data, _, _ = gp.load_all_data_for_graph(G_name)
    X_test = xgboost_data['X_test']
    y_test = xgboost_data['y_test']
    
    pos_sp = X_test[y_test == 1]['sp'].value_counts(normalize=True).to_dict()
    neg_sp = X_test[y_test == 0]['sp'].value_counts(normalize=True).to_dict()
    
    row = {'sbm_param': round(i, 2)}
    for dist in range(1, 7):
        row[f'pos_sp_{dist}'] = pos_sp.get(dist, 0)
        row[f'neg_sp_{dist}'] = neg_sp.get(dist, 0)
    
    row['pos_sp_7+'] = sum(v for k, v in pos_sp.items() if k >= 7 or k == 0)
    row['neg_sp_7+'] = sum(v for k, v in neg_sp.items() if k >= 7 or k == 0)
    
    all_distribs.append(row)

df_sp_detailed = pd.DataFrame(all_distribs)

cols = ['sbm_param'] + [f'{prefix}_sp_{d}' for d in list(range(1, 7)) + ['7+'] for prefix in ['pos', 'neg']]
df_sp_detailed = df_sp_detailed[cols]

pd.options.display.float_format = '{:.3f}'.format
print(df_sp_detailed.to_string(index=False))
"""
all_summaries = []

for i in np.arange(0, 1.25, 0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv4_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    _, _, _, xgboost_data, shap_explainer, _ = gp.load_all_data_for_graph(G_name)
    X_test = xgboost_data['X_test']

    print(f"Type de l'objet : {type(shap_explainer)}")
    print(f"Dimensions des valeurs : {shap_explainer.values.shape}")

    vals = shap_explainer.values
    
    if isinstance(vals, list):
        vals = vals[1]

    df_shap = pd.DataFrame(vals, columns=X_test.columns)
    mean_shap = df_shap.mean()
    mean_abs_shap = df_shap.abs().mean()
    
    summary = pd.DataFrame({
        'feature': X_test.columns,
        'mean_shap': mean_shap.values,
        'mean_abs_shap': mean_abs_shap.values,
        'sbm_param': i
    })
    all_summaries.append(summary)

df_shapvals_concat = pd.concat(all_summaries)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_importance_evolution(df_results, metric="mean_shap", as_ratio=True):
    """
    Génère une heatmap montrant l'évolution de l'importance des features (SHAP).
    """
    heatmap_data = df_results.pivot(index='feature', columns='sbm_param', values=metric)
    
    if as_ratio:
        heatmap_data = heatmap_data.div(heatmap_data.sum(axis=0), axis=1)
        label_name = f"Ratio de contribution ({metric})"
        fmt_type = ".2%"
    else:
        label_name = f"Valeur brute ({metric})"
        fmt_type = ".3f"

    heatmap_data['avg_importance'] = heatmap_data.mean(axis=1)
    heatmap_data = heatmap_data.sort_values(by='avg_importance', ascending=False).drop(columns='avg_importance')

    plt.figure(figsize=(16, 9))
    sns.heatmap(heatmap_data, annot=True, fmt=fmt_type, cmap="YlGnBu", yticklabels=True, cbar_kws={'label': label_name})

    title_suffix = "(Ratios normalisés)" if as_ratio else "(Valeurs brutes)"
    plt.title(f"Importance relative des features (SHAP) selon la structure du graphe {title_suffix}")
    plt.xlabel("Paramètre du graphe (i)")
    plt.ylabel("Features")
    plt.tight_layout()

    filename = f"outputs/plots/heatmap_feat_importance_per_ratio_v4_{metric}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé sous : {filename}")
    plt.show()

plot_feature_importance_evolution(df_shapvals_concat, metric="mean_abs_shap", as_ratio=False)
print(df_shapvals_concat.columns)
features_list = df_shapvals_concat['feature']
print(", ".join(map(str, features_list)))

"""
all_distribs = []
for i in np.arange(0, 1.05, 0.05):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbm_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    _, _,_, xgboost_data, _, _ = gp.load_all_data_for_graph(G_name)
    X_test = xgboost_data['X_test']
    y_test = xgboost_data['y_test']
    
    pos_sp = X_test[y_test == 1]['sp'].value_counts(normalize=True).to_dict()
    neg_sp = X_test[y_test == 0]['sp'].value_counts(normalize=True).to_dict()
    
    row = {'sbm_param': round(i, 2)}
    for dist in range(1, 7):
        row[f'pos_sp_{dist}'] = pos_sp.get(dist, 0)
        row[f'neg_sp_{dist}'] = neg_sp.get(dist, 0)
    
    row['pos_sp_7+'] = sum(v for k, v in pos_sp.items() if k >= 7 or k == 0)
    row['neg_sp_7+'] = sum(v for k, v in neg_sp.items() if k >= 7 or k == 0)
    
    all_distribs.append(row)

df_sp_detailed = pd.DataFrame(all_distribs)

cols = ['sbm_param'] + [f'{prefix}_sp_{d}' for d in list(range(1, 7)) + ['7+'] for prefix in ['pos', 'neg']]
df_sp_detailed = df_sp_detailed[cols]

pd.options.display.float_format = '{:.3f}'.format
print(df_sp_detailed.to_string(index=False))
"""
all_summaries = []

for i in np.arange(0, 1.25, 0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv5_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    _, _, _, xgboost_data, shap_explainer, _ = gp.load_all_data_for_graph(G_name)
    X_test = xgboost_data['X_test']

    print(f"Type de l'objet : {type(shap_explainer)}")
    print(f"Dimensions des valeurs : {shap_explainer.values.shape}")

    vals = shap_explainer.values
    
    if isinstance(vals, list):
        vals = vals[1]

    df_shap = pd.DataFrame(vals, columns=X_test.columns)
    mean_shap = df_shap.mean()
    mean_abs_shap = df_shap.abs().mean()
    
    summary = pd.DataFrame({
        'feature': X_test.columns,
        'mean_shap': mean_shap.values,
        'mean_abs_shap': mean_abs_shap.values,
        'sbm_param': i
    })
    all_summaries.append(summary)

df_shapvals_concat = pd.concat(all_summaries)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_importance_evolution(df_results, metric="mean_shap", as_ratio=True):
    """
    Génère une heatmap montrant l'évolution de l'importance des features (SHAP).
    """
    heatmap_data = df_results.pivot(index='feature', columns='sbm_param', values=metric)
    
    if as_ratio:
        heatmap_data = heatmap_data.div(heatmap_data.sum(axis=0), axis=1)
        label_name = f"Ratio de contribution ({metric})"
        fmt_type = ".2%"
    else:
        label_name = f"Valeur brute ({metric})"
        fmt_type = ".3f"

    heatmap_data['avg_importance'] = heatmap_data.mean(axis=1)
    heatmap_data = heatmap_data.sort_values(by='avg_importance', ascending=False).drop(columns='avg_importance')

    plt.figure(figsize=(16, 9))
    sns.heatmap(heatmap_data, annot=True, fmt=fmt_type, cmap="YlGnBu", yticklabels=True, cbar_kws={'label': label_name})

    title_suffix = "(Ratios normalisés)" if as_ratio else "(Valeurs brutes)"
    plt.title(f"Importance relative des features (SHAP) selon la structure du graphe {title_suffix}")
    plt.xlabel("Paramètre du graphe (i)")
    plt.ylabel("Features")
    plt.tight_layout()

    filename = f"outputs/plots/heatmap_feat_importance_per_ratio_v5_{metric}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé sous : {filename}")
    plt.show()

plot_feature_importance_evolution(df_shapvals_concat, metric="mean_abs_shap", as_ratio=False)
print(df_shapvals_concat.columns)
features_list = df_shapvals_concat['feature']
print(", ".join(map(str, features_list)))

"""
all_distribs = []
for i in np.arange(0, 1.05, 0.05):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbm_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    _, _,_, xgboost_data, _, _ = gp.load_all_data_for_graph(G_name)
    X_test = xgboost_data['X_test']
    y_test = xgboost_data['y_test']
    
    pos_sp = X_test[y_test == 1]['sp'].value_counts(normalize=True).to_dict()
    neg_sp = X_test[y_test == 0]['sp'].value_counts(normalize=True).to_dict()
    
    row = {'sbm_param': round(i, 2)}
    for dist in range(1, 7):
        row[f'pos_sp_{dist}'] = pos_sp.get(dist, 0)
        row[f'neg_sp_{dist}'] = neg_sp.get(dist, 0)
    
    row['pos_sp_7+'] = sum(v for k, v in pos_sp.items() if k >= 7 or k == 0)
    row['neg_sp_7+'] = sum(v for k, v in neg_sp.items() if k >= 7 or k == 0)
    
    all_distribs.append(row)

df_sp_detailed = pd.DataFrame(all_distribs)

cols = ['sbm_param'] + [f'{prefix}_sp_{d}' for d in list(range(1, 7)) + ['7+'] for prefix in ['pos', 'neg']]
df_sp_detailed = df_sp_detailed[cols]

pd.options.display.float_format = '{:.3f}'.format
print(df_sp_detailed.to_string(index=False))
"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_importance_evolution(df_results, metric="mean_shap", as_ratio=True):
    """
    Génère une heatmap montrant l'évolution de l'importance des features (SHAP).
    """
    heatmap_data = df_results.pivot(index='feature', columns='sbm_param', values=metric)
    
    if as_ratio:
        heatmap_data = heatmap_data.div(heatmap_data.sum(axis=0), axis=1)
        label_name = f"Ratio de contribution ({metric})"
        fmt_type = ".2%"
    else:
        label_name = f"Valeur brute ({metric})"
        fmt_type = ".3f"

    heatmap_data['avg_importance'] = heatmap_data.mean(axis=1)
    heatmap_data = heatmap_data.sort_values(by='avg_importance', ascending=False).drop(columns='avg_importance')

    plt.figure(figsize=(16, 9))
    sns.heatmap(heatmap_data, annot=True, fmt=fmt_type, cmap="YlGnBu", yticklabels=True, cbar_kws={'label': label_name})

    title_suffix = "(Ratios normalisés)" if as_ratio else "(Valeurs brutes)"
    plt.title(f"Importance relative des features (SHAP) selon la structure du graphe {title_suffix}")
    plt.xlabel("Paramètre du graphe (i)")
    plt.ylabel("Features")
    plt.tight_layout()

    filename = f"outputs/plots/heatmap_feat_importance_per_ratio_v5_{metric}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé sous : {filename}")
    plt.show()

plot_feature_importance_evolution(df_shapvals_concat, metric="mean_abs_shap", as_ratio=False)
print(df_shapvals_concat.columns)
features_list = df_shapvals_concat['feature']
print(", ".join(map(str, features_list)))

"""
all_distribs = []
for i in np.arange(0, 1.05, 0.05):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbm_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    _, _,_, xgboost_data, _, _ = gp.load_all_data_for_graph(G_name)
    X_test = xgboost_data['X_test']
    y_test = xgboost_data['y_test']
    
    pos_sp = X_test[y_test == 1]['sp'].value_counts(normalize=True).to_dict()
    neg_sp = X_test[y_test == 0]['sp'].value_counts(normalize=True).to_dict()
    
    row = {'sbm_param': round(i, 2)}
    for dist in range(1, 7):
        row[f'pos_sp_{dist}'] = pos_sp.get(dist, 0)
        row[f'neg_sp_{dist}'] = neg_sp.get(dist, 0)
    
    row['pos_sp_7+'] = sum(v for k, v in pos_sp.items() if k >= 7 or k == 0)
    row['neg_sp_7+'] = sum(v for k, v in neg_sp.items() if k >= 7 or k == 0)
    
    all_distribs.append(row)

df_sp_detailed = pd.DataFrame(all_distribs)

cols = ['sbm_param'] + [f'{prefix}_sp_{d}' for d in list(range(1, 7)) + ['7+'] for prefix in ['pos', 'neg']]
df_sp_detailed = df_sp_detailed[cols]

pd.options.display.float_format = '{:.3f}'.format
print(df_sp_detailed.to_string(index=False))
"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_importance_evolution(df_results, metric="mean_abs_shap", as_ratio=True):
    """
    Génère une heatmap montrant l'évolution de l'importance des features (SHAP).
    """
    heatmap_data = df_results.pivot(index='feature', columns='sbm_param', values=metric)
    
    if as_ratio:
        heatmap_data = heatmap_data.div(heatmap_data.sum(axis=0), axis=1)
        label_name = f"Ratio de contribution ({metric})"
        fmt_type = ".2%"
    else:
        label_name = f"Valeur brute ({metric})"
        fmt_type = ".3f"

    heatmap_data['avg_importance'] = heatmap_data.mean(axis=1)
    heatmap_data = heatmap_data.sort_values(by='avg_importance', ascending=False).drop(columns='avg_importance')

    plt.figure(figsize=(16, 9))
    sns.heatmap(heatmap_data, annot=True, fmt=fmt_type, cmap="YlGnBu", yticklabels=True, cbar_kws={'label': label_name})

    title_suffix = "(Ratios normalisés)" if as_ratio else "(Valeurs brutes)"
    plt.title(f"Importance relative des features (SHAP) selon la structure du graphe {title_suffix}")
    plt.xlabel("Paramètre du graphe (i)")
    plt.ylabel("Features")
    plt.tight_layout()

    filename = f"outputs/plots/heatmap_feat_importance_per_ratio_v5_{metric}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé sous : {filename}")
    plt.show()

plot_feature_importance_evolution(df_shapvals_concat, metric="mean_abs_shap", as_ratio=False)
print(df_shapvals_concat.columns)
features_list = df_shapvals_concat['feature']
print(", ".join(map(str, features_list)))

"""
all_distribs = []
for i in np.arange(0, 1.05, 0.05):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbm_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    _, _,_, xgboost_data, _, _ = gp.load_all_data_for_graph(G_name)
    X_test = xgboost_data['X_test']
    y_test = xgboost_data['y_test']
    
    pos_sp = X_test[y_test == 1]['sp'].value_counts(normalize=True).to_dict()
    neg_sp = X_test[y_test == 0]['sp'].value_counts(normalize=True).to_dict()
    
    row = {'sbm_param': round(i, 2)}
    for dist in range(1, 7):
        row[f'pos_sp_{dist}'] = pos_sp.get(dist, 0)
        row[f'neg_sp_{dist}'] = neg_sp.get(dist, 0)
    
    row['pos_sp_7+'] = sum(v for k, v in pos_sp.items() if k >= 7 or k == 0)
    row['neg_sp_7+'] = sum(v for k, v in neg_sp.items() if k >= 7 or k == 0)
    
    all_distribs.append(row)

df_sp_detailed = pd.DataFrame(all_distribs)

cols = ['sbm_param'] + [f'{prefix}_sp_{d}' for d in list(range(1, 7)) + ['7+'] for prefix in ['pos', 'neg']]
df_sp_detailed = df_sp_detailed[cols]

pd.options.display.float_format = '{:.3f}'.format
print(df_sp_detailed.to_string(index=False))
"""
all_summaries = []

for i in np.arange(0, 1.25, 0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv4_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    _, _, _, xgboost_data, shap_explainer, _ = gp.load_all_data_for_graph(G_name)
    X_test = xgboost_data['X_test']

    print(f"Type de l'objet : {type(shap_explainer)}")
    print(f"Dimensions des valeurs : {shap_explainer.values.shape}")

    vals = shap_explainer.values
    
    if isinstance(vals, list):
        vals = vals[1]

    df_shap = pd.DataFrame(vals, columns=X_test.columns)
    mean_shap = df_shap.mean()
    mean_abs_shap = df_shap.abs().mean()
    
    summary = pd.DataFrame({
        'feature': X_test.columns,
        'mean_shap': mean_shap.values,
        'mean_abs_shap': mean_abs_shap.values,
        'sbm_param': i
    })
    all_summaries.append(summary)

df_shapvals_concat = pd.concat(all_summaries)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_importance_evolution(df_results, metric="mean_abs_shap", as_ratio=True):
    """
    Génère une heatmap montrant l'évolution de l'importance des features (SHAP).
    """
    heatmap_data = df_results.pivot(index='feature', columns='sbm_param', values=metric)
    
    if as_ratio:
        heatmap_data = heatmap_data.div(heatmap_data.sum(axis=0), axis=1)
        label_name = f"Ratio de contribution ({metric})"
        fmt_type = ".2%"
    else:
        label_name = f"Valeur brute ({metric})"
        fmt_type = ".3f"

    heatmap_data['avg_importance'] = heatmap_data.mean(axis=1)
    heatmap_data = heatmap_data.sort_values(by='avg_importance', ascending=False).drop(columns='avg_importance')

    plt.figure(figsize=(16, 9))
    sns.heatmap(heatmap_data, annot=True, fmt=fmt_type, cmap="YlGnBu", yticklabels=True, cbar_kws={'label': label_name})

    title_suffix = "(Ratios normalisés)" if as_ratio else "(Valeurs brutes)"
    plt.title(f"Importance relative des features (SHAP) selon la structure du graphe {title_suffix}")
    plt.xlabel("Paramètre du graphe (i)")
    plt.ylabel("Features")
    plt.tight_layout()

    filename = f"outputs/plots/heatmap_feat_importance_per_ratio_v5_{metric}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé sous : {filename}")
    plt.show()

plot_feature_importance_evolution(df_shapvals_concat, metric="mean_abs_shap", as_ratio=False)
print(df_shapvals_concat.columns)
features_list = df_shapvals_concat['feature']
print(", ".join(map(str, features_list)))

"""
all_distribs = []
for i in np.arange(0, 1.05, 0.05):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbm_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    _, _,_, xgboost_data, _, _ = gp.load_all_data_for_graph(G_name)
    X_test = xgboost_data['X_test']
    y_test = xgboost_data['y_test']
    
    pos_sp = X_test[y_test == 1]['sp'].value_counts(normalize=True).to_dict()
    neg_sp = X_test[y_test == 0]['sp'].value_counts(normalize=True).to_dict()
    
    row = {'sbm_param': round(i, 2)}
    for dist in range(1, 7):
        row[f'pos_sp_{dist}'] = pos_sp.get(dist, 0)
        row[f'neg_sp_{dist}'] = neg_sp.get(dist, 0)
    
    row['pos_sp_7+'] = sum(v for k, v in pos_sp.items() if k >= 7 or k == 0)
    row['neg_sp_7+'] = sum(v for k, v in neg_sp.items() if k >= 7 or k == 0)
    
    all_distribs.append(row)

df_sp_detailed = pd.DataFrame(all_distribs)

cols = ['sbm_param'] + [f'{prefix}_sp_{d}' for d in list(range(1, 7)) + ['7+'] for prefix in ['pos', 'neg']]
df_sp_detailed = df_sp_detailed[cols]

pd.options.display.float_format = '{:.3f}'.format
print(df_sp_detailed.to_string(index=False))
"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_importance_evolution(df_results, metric="mean_abs_shap", as_ratio=True):
    """
    Génère une heatmap montrant l'évolution de l'importance des features (SHAP).
    """
    heatmap_data = df_results.pivot(index='feature', columns='sbm_param', values=metric)
    
    if as_ratio:
        heatmap_data = heatmap_data.div(heatmap_data.sum(axis=0), axis=1)
        label_name = f"Ratio de contribution ({metric})"
        fmt_type = ".2%"
    else:
        label_name = f"Valeur brute ({metric})"
        fmt_type = ".3f"

    heatmap_data['avg_importance'] = heatmap_data.mean(axis=1)
    heatmap_data = heatmap_data.sort_values(by='avg_importance', ascending=False).drop(columns='avg_importance')

    plt.figure(figsize=(16, 9))
    sns.heatmap(heatmap_data, annot=True, fmt=fmt_type, cmap="YlGnBu", yticklabels=True, cbar_kws={'label': label_name})

    title_suffix = "(Ratios normalisés)" if as_ratio else "(Valeurs brutes)"
    plt.title(f"Importance relative des features (SHAP) selon la structure du graphe {title_suffix}")
    plt.xlabel("Paramètre du graphe (i)")
    plt.ylabel("Features")
    plt.tight_layout()

    filename = f"outputs/plots/heatmap_feat_importance_per_ratio_v4_{metric}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé sous : {filename}")
    plt.show()

plot_feature_importance_evolution(df_shapvals_concat, metric="mean_abs_shap", as_ratio=False)
print(df_shapvals_concat.columns)
features_list = df_shapvals_concat['feature']
print(", ".join(map(str, features_list)))

"""
all_distribs = []
for i in np.arange(0, 1.05, 0.05):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbm_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    _, _,_, xgboost_data, _, _ = gp.load_all_data_for_graph(G_name)
    X_test = xgboost_data['X_test']
    y_test = xgboost_data['y_test']
    
    pos_sp = X_test[y_test == 1]['sp'].value_counts(normalize=True).to_dict()
    neg_sp = X_test[y_test == 0]['sp'].value_counts(normalize=True).to_dict()
    
    row = {'sbm_param': round(i, 2)}
    for dist in range(1, 7):
        row[f'pos_sp_{dist}'] = pos_sp.get(dist, 0)
        row[f'neg_sp_{dist}'] = neg_sp.get(dist, 0)
    
    row['pos_sp_7+'] = sum(v for k, v in pos_sp.items() if k >= 7 or k == 0)
    row['neg_sp_7+'] = sum(v for k, v in neg_sp.items() if k >= 7 or k == 0)
    
    all_distribs.append(row)

df_sp_detailed = pd.DataFrame(all_distribs)

cols = ['sbm_param'] + [f'{prefix}_sp_{d}' for d in list(range(1, 7)) + ['7+'] for prefix in ['pos', 'neg']]
df_sp_detailed = df_sp_detailed[cols]

pd.options.display.float_format = '{:.3f}'.format
print(df_sp_detailed.to_string(index=False))
"""
all_summaries = []

for i in np.arange(0, 1.25, 0.25):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv5_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    _, _, _, xgboost_data, shap_explainer, _ = gp.load_all_data_for_graph(G_name)
    X_test = xgboost_data['X_test']

    print(f"Type de l'objet : {type(shap_explainer)}")
    print(f"Dimensions des valeurs : {shap_explainer.values.shape}")

    vals = shap_explainer.values
    
    if isinstance(vals, list):
        vals = vals[1]

    df_shap = pd.DataFrame(vals, columns=X_test.columns)
    mean_shap = df_shap.mean()
    mean_abs_shap = df_shap.abs().mean()
    
    summary = pd.DataFrame({
        'feature': X_test.columns,
        'mean_shap': mean_shap.values,
        'mean_abs_shap': mean_abs_shap.values,
        'sbm_param': i
    })
    all_summaries.append(summary)

df_shapvals_concat = pd.concat(all_summaries)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_importance_evolution(df_results, metric="mean_abs_shap", as_ratio=True):
    """
    Génère une heatmap montrant l'évolution de l'importance des features (SHAP).
    """
    heatmap_data = df_results.pivot(index='feature', columns='sbm_param', values=metric)
    
    if as_ratio:
        heatmap_data = heatmap_data.div(heatmap_data.sum(axis=0), axis=1)
        label_name = f"Ratio de contribution ({metric})"
        fmt_type = ".2%"
    else:
        label_name = f"Valeur brute ({metric})"
        fmt_type = ".3f"

    heatmap_data['avg_importance'] = heatmap_data.mean(axis=1)
    heatmap_data = heatmap_data.sort_values(by='avg_importance', ascending=False).drop(columns='avg_importance')

    plt.figure(figsize=(16, 9))
    sns.heatmap(heatmap_data, annot=True, fmt=fmt_type, cmap="YlGnBu", yticklabels=True, cbar_kws={'label': label_name})

    title_suffix = "(Ratios normalisés)" if as_ratio else "(Valeurs brutes)"
    plt.title(f"Importance relative des features (SHAP) selon la structure du graphe {title_suffix}")
    plt.xlabel("Paramètre du graphe (i)")
    plt.ylabel("Features")
    plt.tight_layout()

    filename = f"outputs/plots/heatmap_feat_importance_per_ratio_v5_{metric}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé sous : {filename}")
    plt.show()

plot_feature_importance_evolution(df_shapvals_concat, metric="mean_abs_shap", as_ratio=False)
print(df_shapvals_concat.columns)
features_list = df_shapvals_concat['feature']
print(", ".join(map(str, features_list)))

"""
all_distribs = []
for i in np.arange(0, 1.05, 0.05):
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbm_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    
    _, _,_, xgboost_data, _, _ = gp.load_all_data_for_graph(G_name)
    X_test = xgboost_data['X_test']
    y_test = xgboost_data['y_test']
    
    pos_sp = X_test[y_test == 1]['sp'].value_counts(normalize=True).to_dict()
    neg_sp = X_test[y_test == 0]['sp'].value_counts(normalize=True).to_dict()
    
    row = {'sbm_param': round(i, 2)}
    for dist in range(1, 7):
        row[f'pos_sp_{dist}'] = pos_sp.get(dist, 0)
        row[f'neg_sp_{dist}'] = neg_sp.get(dist, 0)
    
    row['pos_sp_7+'] = sum(v for k, v in pos_sp.items() if k >= 7 or k == 0)
    row['neg_sp_7+'] = sum(v for k, v in neg_sp.items() if k >= 7 or k == 0)
    
    all_distribs.append(row)

df_sp_detailed = pd.DataFrame(all_distribs)

cols = ['sbm_param'] + [f'{prefix}_sp_{d}' for d in list(range(1, 7)) + ['7+'] for prefix in ['pos', 'neg']]
df_sp_detailed = df_sp_detailed[cols]

pd.options.display.float_format = '{:.3f}'.format
print(df_sp_detailed.to_string(index=False))
"""
gt_commu = 'GT_sbm_density'
gt_spatial = 'GT_pos_dist'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

def add_labels_inverted(ax, data, hue_col, x_target, alignment='left'):
    lines = ax.get_lines()
    features = data[hue_col].unique()
    for i, feature in enumerate(features):
        subset = data[data[hue_col] == feature]
        idx = (subset['alpha'] - x_target).abs().idxmin()
        point = subset.loc[idx]
        color = lines[i].get_color()
        offset = 0.02 if alignment == 'left' else -0.02
        ax.text(point['alpha'] + offset, point['correlation'], feature, fontsize=8, va='center', ha=alignment, color=color, fontweight='bold', alpha=0.7)

# --- Plot 1 : SBM (Labels à DROITE @ alpha=1.0) ---
data_commu = df_evolution[df_evolution['gt_feature'] == gt_commu]
sns.lineplot(data=data_commu, x='alpha', y='correlation', hue='feature', marker='o', ax=ax1)
ax1.set_title(f"Affinité Communautaire\n(Graphe pur sbm pour $\\alpha=1$)", fontsize=13)
add_labels_inverted(ax1, data_commu, 'feature', x_target=1.0, alignment='left')
ax1.set_xlim(-0.05, 1.1) 
ax1.legend(title="Features", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='x-small', borderaxespad=0.)

# --- Plot 2 : Spatial (Labels à GAUCHE @ alpha=0.0) ---
data_spatial = df_evolution[df_evolution['gt_feature'] == gt_spatial]
sns.lineplot(data=data_spatial, x='alpha', y='correlation', hue='feature', marker='o', ax=ax2, legend=False)
ax2.set_title(f"Affinité Spatiale\n(Graphe pur spatial pour $\\alpha=0$)", fontsize=13)
add_labels_inverted(ax2, data_spatial, 'feature', x_target=0.0, alignment='right')
ax2.set_xlim(-0.1, 1.05)

for ax in [ax1, ax2]:
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlabel("Alpha (0=Spatial, 1=SBM)", fontweight='bold')
    ax.set_ylabel("Corrélation de Spearman", fontweight='bold')
    ax2.tick_params(labelleft=True)
    ax2.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    
plt.subplots_adjust(wspace=0.18) 
plt.show()
gt_commu = 'GT_sbm_density'
gt_spatial = 'GT_pos_dist'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

def add_labels_inverted(ax, data, hue_col, x_target, alignment='left'):
    lines = ax.get_lines()
    features = data[hue_col].unique()
    for i, feature in enumerate(features):
        subset = data[data[hue_col] == feature]
        idx = (subset['alpha'] - x_target).abs().idxmin()
        point = subset.loc[idx]
        color = lines[i].get_color()
        offset = 0.02 if alignment == 'left' else -0.02
        ax.text(point['alpha'] + offset, point['correlation'], feature, fontsize=8, va='center', ha=alignment, color=color, fontweight='bold', alpha=0.7)

# --- Plot 1 : SBM (Labels à DROITE @ alpha=1.0) ---
data_commu = df_evolution[df_evolution['gt_feature'] == gt_commu]
sns.lineplot(data=data_commu, x='alpha', y='correlation', hue='feature', marker='o', ax=ax1)
ax1.set_title(f"Affinité Communautaire\n(Corrélation avec {gt_commu})", fontsize=13)
add_labels_inverted(ax1, data_commu, 'feature', x_target=1.0, alignment='left')
ax1.set_xlim(-0.05, 1.1) 
ax1.legend(title="Features", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='x-small', borderaxespad=0.)

# --- Plot 2 : Spatial (Labels à GAUCHE @ alpha=0.0) ---
data_spatial = df_evolution[df_evolution['gt_feature'] == gt_spatial]
sns.lineplot(data=data_spatial, x='alpha', y='correlation', hue='feature', marker='o', ax=ax2, legend=False)
ax2.set_title(f"Affinité Spatiale\n(Corrélation avec {gt_spatial})", fontsize=13)
add_labels_inverted(ax2, data_spatial, 'feature', x_target=0.0, alignment='right')
ax2.set_xlim(-0.1, 1.05)

for ax in [ax1, ax2]:
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlabel("Alpha (0=Spatial, 1=SBM)", fontweight='bold')
    ax.set_ylabel("Corrélation de Spearman", fontweight='bold')
    ax2.tick_params(labelleft=True)
    ax2.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    
plt.subplots_adjust(wspace=0.18) 
plt.show()
import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration des itérations
steps = np.arange(0, 1.25, 0.25)

for i in steps:
    sbm_val = f"{i:.2f}"
    pos_val = f"{1-i:.2f}"
    G_name = f"artificial_graph_sbmv2_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}"
    file_path = f"graph_library/{G_name}.json"
    
    if os.path.exists(file_path):
        # 1. Chargement du graphe
        with open(file_path, 'r') as f:
            data = json.load(f)
            # On utilise node_link_graph pour le format JSON standard de NetworkX
            G = nx.readwrite.json_graph.node_link_graph(data)
        
        # 2. Calcul de la distribution des degrés
        degrees = [d for n, d in G.degree()]
        
        # 3. Plot
        plt.figure(figsize=(8, 4))
        plt.hist(degrees, bins=range(min(degrees), max(degrees) + 2), 
                 color='skyblue', edgecolor='black', alpha=0.7)
        
        plt.title(f"Distribution des degrés : {G_name}")
        plt.xlabel("Degré")
        plt.ylabel("Nombre de nœuds")
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.show()
    else:
        print(f"Fichier non trouvé : {file_path}")
%history -f recuperation_code.py
