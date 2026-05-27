from .utils import *
from .models import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
from joblib import Parallel, delayed
    
def execute(G, G_name, add_P_matrix = False, steps= ["prep", "shap"]): 

    if 'prep' in steps:
        validate_input_graph(G)
        print("[PREP] Validation du Graphe terminée. Lancement des calculs...")

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

        if GT is not None and 'GT_pos' in GT:
            for i, node_id in enumerate(G.nodes()):
                G.nodes[node_id]['GT_pos'] = GT['GT_pos'][i]

       
        G_kept, G_hidden = hide_graph_links(G, test_size=0.10)
        G_train, G_test = hide_graph_links(G_kept, test_size=0.15)
        loadsave_data_joblib(data=G_kept, filename=f"G_train_init_{G_name}", mode="save")
    
        best_params, results_summary = k_fold_cross_validation(G_kept, k=1, features_list=None, n_trials=30, GroundTruth=GT, graph_name= G_name)
        print(best_params)

        G_train_with_structure = computeStructureFeatures(G_train)
        G_train_with_distances = computeDistanceFeatures(G_train_with_structure)
        G_train_with_communities = computeCommunityFeatures(G_train_with_distances)

        G_kept_with_structure = computeStructureFeatures(G_kept)
        G_kept_with_distances = computeDistanceFeatures(G_kept_with_structure)
        G_kept_with_communities = computeCommunityFeatures(G_kept_with_distances)
        
        print("Sauvegarde des datasets")
        loadsave_data_joblib(data=G_train_with_communities, filename=f"G_train_w_struct_com_dist_{G_name}", mode="save")
        loadsave_data_joblib(data=G_kept_with_communities, filename=f"G_kept_w_struct_com_dist_{G_name}", mode="save")
    
        dataset_train = prepare_balanced_data(G_test, G_train_with_communities,  negative_ratio=10.0, GroundTruth=GT,)
        dataset_hidden = prepare_balanced_data(G_hidden, G_kept_with_communities, negative_ratio=50.0, GroundTruth=GT,)
    
        print("Vérif : colonnes du dataset :")
        print(dataset_train.columns)
        save_dataset(dataset=dataset_train, filename=f"dataset_train_{G_name}")
        save_dataset(dataset=dataset_hidden, filename=f"dataset_hidden_{G_name}")

        # Choix des features pour l'execution finale
        exclude = ['u', 'v', 'target', 'label'] 
        features = [
            col for col in dataset_train.columns 
            if (col not in exclude and not col.startswith('GT_'))
            #or col in ['GT_sbm_density', 'GT_pos_dist','GT_spatial_deg_product', 'GT_sbm_deg_product']
        ]
        
        results_test, model, X_train, y_train, X_test, y_test = train_and_test_xgboost(dataset_train, features=features, parameters=best_params)
        
        X_hidden = dataset_hidden[features] if features else dataset_hidden.drop(["target", "u", "v", "label"], axis=1)
        y_hidden = dataset_hidden['target']
        
        results_hidden = get_performance_metrics(model, X_hidden, y_hidden, "Hidden_")
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
        loadsave_data_joblib(data=data_to_save, filename=f"xgboost_data_{G_name}.joblib", mode="save")
    
        print("\n RÉSULTATS")
        print(results_test_hidden.to_string(index=False))

    if 'shap' in steps:
        print("\n [SHAP] Shapley va ! Lu.")
    
        if 'prep' not in steps:
            print(f" Chargement des données pré-calculées...")
            data = loadsave_data_joblib(filename=f"xgboost_data_{G_name}.joblib", mode="load")
            model = data['model']
            X_hidden = data['X_hidden']
            y_hidden = data['y_hidden']

        shap_explanation = analyze_with_shap_tree(model, X_hidden, y_hidden)
        print("Sauvegarde de l'analyse SHAP")
        loadsave_data_joblib(data=shap_explanation, filename=f"shap_explainer_{G_name}.joblib", mode="save")

def evaluate(G_name, display=False):
    shap_base = loadsave_data_joblib(data=None, filename=f"shap_explainer_{G_name}.joblib", mode="load")
    xgboost_data = loadsave_data_joblib(data=None, filename=f"xgboost_data_{G_name}.joblib", mode="load")
    
    group_mapping = {
        "Groupe_Structure": ['cn', 'aa', 'jc', 'pa', 'sp', 'pr_u', 'pr_v', 'lcc_u', 'lcc_v', 'and_u', 'and_v', 'dc_u', 'dc_v'],
        "Groupe_Communities": ['sbm_density', 'same_sbm', 'infomap_density', 'same_infomap',"louvain_density", "same_louvain"],
        "Groupe_Embeddings": ['n2v_homophily_cos', 'n2v_homophily_dist', 'deepwalk_cos', 'deepwalk_dist']
    }

    group_names = list(group_mapping.keys())

    # --- ÉTAPE A & B : APPROCHES PAR SOMME (HEURISTIQUE) ---
    group_col_indices = [
        [shap_base.feature_names.index(f) for f in features]
        for features in group_mapping.values()
    ]

    # Somme signée (conserve les directions positive/négative)
    val_aggr_sum = np.array([shap_base.values[:, idxs].sum(axis=1) for idxs in group_col_indices]).T
    # Somme des valeurs absolues (mesure l'effort total fourni dans le groupe)
    val_aggr_abs = np.array([np.abs(shap_base.values[:, idxs]).sum(axis=1) for idxs in group_col_indices]).T

    # Métriques de conflit interne (Contradiction)
    # Mesure si les variables d'un même groupe s'annulent entre elles
    conflict_ratio = (val_aggr_abs - np.abs(val_aggr_sum)) / (val_aggr_abs + 1e-10)
    mean_conflict_ratios = np.mean(conflict_ratio, axis=0)

    # Création des objets Explanation pour l'approche Heuristique
    exp_aggr_sum = shap.Explanation(
        values=val_aggr_sum,
        base_values=shap_base.base_values,
        feature_names=group_names,
    )

    exp_aggr_abs = shap.Explanation(
        values=val_aggr_abs,
        base_values=shap_base.base_values,
        feature_names=group_names,
    )

    # --- ÉTAPE C : APPROCHE PAR COALITION (RECALCUL EXACT) ---
    # On recalcule les valeurs de Shapley en traitant les groupes comme des blocs atomiques
    df_coalition_values = analyse_with_shap_custom(
        model=xgboost_data["model"], 
        X_eval=xgboost_data["X_hidden"], 
        X_train=xgboost_data["X_train"]
    )
    
    # Calcul de la vraie probabilité moyenne du modèle (Baseline)
    train_baseline = xgboost_data["model"].predict_proba(xgboost_data["X_train"])[:, 1].mean()

    exp_coalition_exact = shap.Explanation(
        values=df_coalition_values.values,
        base_values=np.array([train_baseline] * len(df_coalition_values)), 
        feature_names=group_names,
    )

    # --- SAUVEGARDE ---
    results_to_save = {
        "G_name": G_name,
        "group_mapping": group_mapping,
        "mean_conflict_ratios": mean_conflict_ratios,
        "exp_aggr_sum": exp_aggr_sum,
        "exp_aggr_abs": exp_aggr_abs,
        "df_coalition_values": df_coalition_values,
        "exp_coalition_exact": exp_coalition_exact
    }
    
    loadsave_data_joblib(
        data=results_to_save, 
        filename=f"shap_analysis_{G_name}.joblib", 
        mode="save"
    )

    # --- AFFICHAGE ---
    if display:
        print(f"\n=== Analyse SHAP par Groupes : {G_name} ===")
        
        print("\n[1] Approche Somme (Heuristique) - Impact Directionnel")
        shap.plots.bar(exp_aggr_sum)
        
        print("\n[2] Approche Somme (Heuristique) - Importance Totale")
        shap.plots.bar(exp_aggr_abs)
        
        print("\n[3] Approche Coalition (Exacte) - Prise en compte des interactions")
        shap.plots.bar(exp_coalition_exact)

        print(f"\n{'Groupe':<25} | {'Conflict Ratio (Interne)':<20}")
        print("-" * 50)
        for i, name in enumerate(group_names):
            print(f"{name:<25} | {mean_conflict_ratios[i]:.4f}")
            
    return results_to_save
    

def plot_shap_evolution():

    ratios = [round(r, 2) for r in np.linspace(0, 1, 5)]
    valid_ratios = []

    all_results = {
        "base": {"Groupe_Structure": [], "Groupe_Communities": [], "Groupe_Embeddings": []},
        "abs": {"Groupe_Structure": [], "Groupe_Communities": [], "Groupe_Embeddings": []},
        "custom": {"Groupe_Structure": [], "Groupe_Communities": [], "Groupe_Embeddings": []}
    }

    for r in ratios:
        G_name = f"artificial_graph_sbmv2_{r:.2f}_pos_{1-r:.2f}".replace('.', '_')
        print(f"ZIZI MOU {G_name}")
        filename = f"shap_analysis_{G_name}.joblib"
        try:
            data = loadsave_data_joblib(data=None, filename=filename, mode="load")
            
            shaps = {
                "base": data["exp_aggr_sum"],
                "abs": data["exp_aggr_abs"],
                "custom": data["exp_coalition_exact"] 
            }

            for k, exp in shaps.items():
                importances = np.mean(np.abs(exp.values) if k != "abs" else exp.values, axis=0)
                for idx, cat_name in enumerate(exp.feature_names):
                    if cat_name in all_results[k]:
                        all_results[k][cat_name].append(importances[idx])
                    else : 
                        print(f"Catégorie {cat_name} non connue :(")
            
            valid_ratios.append(r)
            
        except FileNotFoundError:
            print("Fichier non trouvé")
            continue

    # --- GÉNÉRATION DES GRAPHIQUES (3 Subplots) ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True)
    
    # Titre général
    fig.suptitle("Évolution de l'importance SHAP selon le ratio SBM", fontsize=16, y=1.02)
    
    titles = {
        "base": "SHAP Normal (Signed Sum)", 
        "abs": "SHAP Absolute (Sum of Abs)", 
        "custom": "SHAP Custom Analysis"
    }
    
    # Couleurs et marqueurs mis à jour avec la bonne clé
    colors = {
        'Groupe_Structure': '#1f77b4', 
        'Groupe_Communities': '#ff7f0e', 
        'Groupe_Embeddings': '#2ca02c'
    }
    
    markers = {
        'Groupe_Structure': 'o', 
        'Groupe_Communities': 's', 
        'Groupe_Embeddings': '^'
    }

    for i, (metrique, ax) in enumerate(zip(["base", "abs", "custom"], axes)):
        # On trace chaque catégorie de features
        for cat_name, values in all_results[metrique].items():
            if len(values) == len(valid_ratios): # Sécurité pour les données manquantes
                ax.plot(
                    valid_ratios, 
                    values, 
                    label=cat_name, 
                    color=colors.get(cat_name, 'gray'), 
                    marker=markers.get(cat_name, 'x'), 
                    linewidth=2,
                    markersize=6
                )
        
        # Style de chaque subplot
        ax.set_title(titles[metrique], fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel("Ratio SBM", fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        if i == 0:
            ax.set_ylabel("Importance Moyenne (SHAP)", fontsize=12)
        
        ax.legend(fontsize=9, loc='best')

    # Ajustement automatique de l'espacement
    plt.tight_layout()
    
    # Sauvegarde
    save_path = "outputs/plots/detailed_shap_evolution.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"\nGraphique sauvegardé avec succès dans : {save_path}")
    
    plt.show()

def compute_commus(G, G_name, spatial_ref = "GT_pos", computeEmb=False):

    validate_input_graph(G)
    print("[PREP] Validation du Graphe terminée. Lancement des calculs...")

    if 'GroundTruth_JSON' in G.graph:
        print(f"[INIT] Extraction de la GT GroundTruth pour {G_name}...")
        gt_raw = json.loads(G.graph['GroundTruth_JSON'])
        raw_pos = gt_raw.get('GT_pos', [])
        gt_pos_array = np.array(raw_pos, dtype=float)

        """
        GT = {'GT_sbm_matrix': np.array(gt_raw['GT_sbm_matrix']),
            'GT_pos': np.array(gt_raw['GT_pos']),
            'GT_sbm_id': np.array(gt_raw['GT_sbm_id']),
            'GT_degrees_sbm': np.array(gt_raw['GT_degrees_sbm']),
            'GT_degrees_spatial': np.array(gt_raw['GT_degrees_spatial'])
                }
        """
        GT = {'GT_pos': gt_pos_array}
                
        if 'P_matrix_JSON' in G.graph:
            print("P_matrix trouvée.")
            GT['GT_proba'] = np.array(json.loads(G.graph['P_matrix_JSON']))
    else:
        print("[WARNING] Aucune GroundTruth_JSON trouvée dans G.graph")
        GT = None

    if GT is not None and 'GT_pos' in GT:
            for i, node_id in enumerate(G.nodes()):
                G.nodes[node_id]['GT_pos'] = GT['GT_pos'][i]
    
    G_kept, G_hidden = hide_graph_links(G, test_size=0.10)
    G_train, G_test = hide_graph_links(G_kept, test_size=0.15)

    if spatial_ref == "deepwalk": 
        G_train = computeDistanceFeatures(G_train)
        G_kept = computeDistanceFeatures(G_kept)
        
    G_train_with_communities = computeCommunityFeatures(G_train, spatial_ref=spatial_ref)
    G_kept_with_communities = computeCommunityFeatures(G_kept, spatial_ref=spatial_ref)
    if computeEmb:
        G_train_with_communities = computeDistanceFeatures(G_train)
        G_kept_with_communities = computeDistanceFeatures(G_kept)
        

    print("Save : graph pour exploration commus")
    loadsave_data_joblib(data=G_kept_with_communities, filename=f"G_kept_w_struct_com_dist_{G_name}", mode="save")
    
    
    dataset_train = prepare_balanced_data(G_test, G_train_with_communities,  negative_ratio=10.0, GroundTruth=GT)
    dataset_hidden = prepare_balanced_data(G_hidden, G_kept_with_communities, negative_ratio=50.0, GroundTruth=GT)

    print("Vérif : colonnes du dataset :")
    print(dataset_train.columns)

    print("Sauvegarde des datasets")
    save_dataset(dataset=dataset_train, filename=f"dataset_train_{G_name}")
    save_dataset(dataset=dataset_hidden, filename=f"dataset_hidden_{G_name}")
    

def analyze_commus(G_name_short, nb_iterations, spatial_ref = "GT_pos", i_min =0.00, i_max = 1.00, nb_i=11, name_export_results="DATE"):
    """
    features_GT_proba = ['GT_proba']
    features_GT_pos = ['GT_pos_dist', 
                       #'GT_spatial_deg_product', 
                    #'GT_spatial_gravity_log', 'GT_degrees_spatial_u','GT_degrees_spatial_v'
                    ]
    features_commu_inferee_normal = ["louvain_density"]
    features_commu_inferee_spatial_based_manual_iter = ["spatial_louvain_density"]
    features_commu_inferee_spatial_based_manual_iter_old = ["spatial_louvain_old_density"]
    features_commu_inferee_spatial_based_manual_iter_bined = ["spatial_louvain_bined_density"]
    features_commu_inferee_spatial_based_manual_iter_0_20 = ["spatial_louvain_manualiter_0_20_density"]
    features_commu_inferee_spatial_based_manual_iter_0_50 = ["spatial_louvain_manualiter_0_50_density"]
    features_commu_inferee_spatial_based_manual_iter_0_80 = ["spatial_louvain_manualiter_0_80_density"]
    features_commu_inferee_spatial_based_manual_iter_0_90 = ["spatial_louvain_manualiter_0_90_density"]
    features_commu_inferee_spatial_based_manual_reg = ["spatial_louvain_manualreg_density"]
    features_commu_inferee_spatial_based_scgravity = ["spatial_louvain_scgravity_density"]
    features_commu_inferee_spatial_based_wrdb = ["spatial_louvain_wrdb_density"]
    features_deepwalk = ["deepwalk_dist"]
    features_SiNEcustom = ["SiNEcustom_dist"]
    features_SiNEcustom_spatial = ["SiNEcustom_spatial_dist"]
    features_SiNEcustom_spatial_bined = ["SiNEcustom_spatial_bined_dist"]
    features_SiNE = ["SiNE_dist"]
    features_SiNE_spatial = ["SiNE_spatial_dist"]
    features_SiNE_spatial_bined = ["SiNE_spatial_bined_dist"]

    experiments = {
        "Inferred_Commu_normal": features_commu_inferee_normal,
        "Inferred_Commu_spatial_manuel_iter": features_commu_inferee_spatial_based_manual_iter,
         "Inferred_Commu_spatial_manuel_iter_old": features_commu_inferee_spatial_based_manual_iter_old,
        "Inferred_Commu_spatial_manuel_iter_bined":features_commu_inferee_spatial_based_manual_iter_bined,
        "Inferred_Commu_spatial_manuel_iter_0_20": features_commu_inferee_spatial_based_manual_iter_0_20,
        "Inferred_Commu_spatial_manuel_iter_0_50": features_commu_inferee_spatial_based_manual_iter_0_50,
        "Inferred_Commu_spatial_manuel_iter_0_80": features_commu_inferee_spatial_based_manual_iter_0_80,
        #"Inferred_Commu_spatial_manuel_iter_0_90": features_commu_inferee_spatial_based_manual_iter_0_90,
        #"Inferred_Commu_spatial_manuel_reg": features_commu_inferee_spatial_based_manual_reg,
        #"Inferred_Commu_spatial_scgravity": features_commu_inferee_spatial_based_scgravity,
        #"Inferred_Commu_spatial_wrdb": features_commu_inferee_spatial_based_wrdb,
        #"GT_proba": features_GT_proba,
        "GT_pos": features_GT_pos,
        "GT_pos + Inferred_Commu normal": features_GT_pos + features_commu_inferee_normal,
        "GT_pos + Inferred_Commu spatial manuel iter": features_GT_pos + features_commu_inferee_spatial_based_manual_iter,
        "GT_pos + Inferred_Commu spatial manuel iter old": features_GT_pos + features_commu_inferee_spatial_based_manual_iter_old,
        "GT_pos + Inferred_Commu spatial manuel iter bined": features_GT_pos + features_commu_inferee_spatial_based_manual_iter_bined,
        "GT_pos + Inferred_Commu spatial manuel iter 0_20": features_GT_pos + features_commu_inferee_spatial_based_manual_iter_0_20,
        "GT_pos + Inferred_Commu spatial manuel iter 0_50": features_GT_pos + features_commu_inferee_spatial_based_manual_iter_0_50,
        "GT_pos + Inferred_Commu spatial manuel iter 0_80": features_GT_pos + features_commu_inferee_spatial_based_manual_iter_0_80,
        #"GT_pos + Inferred_Commu spatial manuel iter 0_90": features_GT_pos + features_commu_inferee_spatial_based_manual_iter_0_90,
        #"GT_pos + Inferred_Commu spatial manuel reg": features_GT_pos + features_commu_inferee_spatial_based_manual_reg,
       # "GT_pos + Inferred_Commu spatial scgravity": features_GT_pos + features_commu_inferee_spatial_based_scgravity,
        #"GT_pos + Inferred_Commu spatial wrdb": features_GT_pos + features_commu_inferee_spatial_based_wrdb,
        #"Deepwalk" : features_deepwalk,
        #"Deepwalk + Inferred_Commu normal": features_deepwalk + features_commu_inferee_normal,
        #"Deepwalk + Inferred_Commu spatial manuel iter": features_deepwalk + features_commu_inferee_spatial_based_manual_iter,
        #"Deepwalk + Inferred_Commu spatial manuel reg": features_deepwalk + features_commu_inferee_spatial_based_manual_reg,
        #"Deepwalk + Inferred_Commu spatial scgravity": features_deepwalk + features_commu_inferee_spatial_based_scgravity,
        #"Deepwalk + Inferred_Commu spatial wrdb": features_deepwalk + features_commu_inferee_spatial_based_wrdb,
        #"SiNEcustom": features_SiNEcustom,
        "SiNEcustom_spatial": features_SiNEcustom_spatial,
        "SiNEcustom_spatial_bined": features_SiNEcustom_spatial_bined,
        #"SiNE": features_SiNE,
        "SiNE_spatial": features_SiNE_spatial,
        "SiNE_spatial_bined": features_SiNE_spatial_bined,
        "deepwalk": features_deepwalk,
        #"GT_pos + SiNEcustom": features_GT_pos + features_SiNEcustom,
        "GT_pos + SiNEcustom_spatial": features_GT_pos + features_SiNEcustom_spatial,
        "GT_pos + SiNEcustom_spatial_bined": features_GT_pos + features_SiNEcustom_spatial_bined,
        #"GT_pos + SiNE": features_GT_pos + features_SiNE,
        "GT_pos + SiNE_spatial": features_GT_pos + features_SiNE_spatial,
        "GT_pos + SiNE_spatial_bined": features_GT_pos + features_SiNE_spatial_bined,
        "GT_pos + deepwalk": features_GT_pos + features_deepwalk,
    }
    """
    features_GT_proba = ['GT_proba']

    experiments = {
        "GT_proba": features_GT_proba,
        
    }

    all_results = []

    tasks = [
        (nb_iter, i) 
        for nb_iter in range(1, nb_iterations + 1) 
        for i in np.linspace(i_max, i_min, nb_i)
    ]

    cores_to_use = max(1, os.cpu_count() -2)

    print(f"Lancement de la parallélisation sur {cores_to_use} coeurs pour {len(tasks)} tâches...")

    # Exécution parallèle
    results_nested = Parallel(n_jobs=cores_to_use)(
        delayed(run_single_experiment)(nb_iter, i, spatial_ref, G_name_short, experiments) 
        for nb_iter, i in tasks
    )


    # Aplatir la liste de listes
    all_results = [item for sublist in results_nested for item in sublist]

    all_results = pd.DataFrame(all_results)
    
    output_dir = "outputs/results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"comparaison_perfs_commus_{G_name_short}_{nb_iterations}iter_{name_export_results}.csv")
    all_results.to_csv(output_path, index=False)
    print(f" Succès ! Fichier sauvegardé dans : {output_path}")
    return all_results

def run_single_experiment(nb_iter, i, spatial_ref, G_name_short, experiments):
        """
        Fonction exécutée par un cœur unique pour une valeur de i et une itération donnée.
        """
        sbm_val = f"{i:.2f}"
        pos_val = f"{1-i:.2f}"
        if spatial_ref == "GT_Pos" or spatial_ref == "GT_pos" : 
            spatial_ref = ""
        else :
            spatial_ref = f"_{spatial_ref}"
        G_name = f"{G_name_short}_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}_{nb_iter}{spatial_ref}"
    

        # 1. Chargement des données d'entraînement
        _, dataset_train, dataset_eval, _, _, _ = load_all_data_for_graph(G_name)
        local_results = []

        for exp_name, feat_list in experiments.items():
            missing = set(feat_list) - set(dataset_train.columns)
            if missing:
                print(f" Exp {exp_name} : colonnes manquantes {missing}. Skip.")
                print(set(dataset_train.columns))
                continue

            print(f" Running: {exp_name} for SBM={i}")
        
            Params = {
                'max_depth': 3,             # Faible profondeur pour éviter l'overfitting sur 2 variables
                'learning_rate': 0.1,       # Compromis idéal vitesse/précision
                'n_estimators': 1000,       # On met beaucoup, l'early stopping fera le reste
                'subsample': 1.0,           # On garde 100% des lignes (plus stable pour peu de features)
                'colsample_bytree': 1.0,    # On garde les 2 features à chaque split
                'objective': 'binary:logistic', 
                'tree_method': 'hist',      # Accélère l'entraînement sur de gros datasets
                'reg_lambda': 1,            # Régularisation L2 pour stabiliser les poids
                'n_jobs': 1                # Utilise 1 seul coeur, pour la parallélisation
            }

            stats_df, model, _, _, _, _ = train_and_test_xgboost(dataset_train, features=feat_list, parameters = Params, plot=False)

            importances = model.feature_importances_
            feat_imp_series = pd.Series(importances, index=feat_list).sort_values(ascending=False)
                
            # Évaluation sur le dataset de référence FIXE (Graphe SBM 1.0)
            X_eval_fixed = dataset_eval[feat_list] 
            stats_eval_df = get_performance_metrics(model, X_eval_fixed, dataset_eval["target"], "EXP_")
            
            local_results.append({
                "G_name" : G_name,
                "Ratio_SBM": i,
                "Iter": nb_iter,
                "Experiment": exp_name,
                "AP_train": stats_df["Test_AP"].iloc[0],
                "AUC-ROC_train": stats_df["Test_AUC-ROC"].iloc[0],
                "AP_eval": stats_eval_df["EXP_AP"].iloc[0],
                "AUC-ROC_eval": stats_eval_df["EXP_AUC-ROC"].iloc[0],
                "Top_Feature": feat_imp_series.index[0], # On stocke la #1 pour analyse
                "Top_Importance": feat_imp_series.iloc[0]
            })

        return local_results

def compute_commus_greels(G, G_name, spatial_ref = "GT_pos"):

    validate_input_graph(G)
    print("[PREP] Validation du Graphe terminée. Lancement des calculs...")

    if 'GroundTruth_JSON' in G.graph:
        print(f"[INIT] Extraction de la GT GroundTruth pour {G_name}...")
        gt_raw = json.loads(G.graph['GroundTruth_JSON'])
        raw_pos = gt_raw.get('GT_pos', [])
        gt_pos_array = np.array(raw_pos, dtype=float)

        """
        GT = {'GT_sbm_matrix': np.array(gt_raw['GT_sbm_matrix']),
            'GT_pos': np.array(gt_raw['GT_pos']),
            'GT_sbm_id': np.array(gt_raw['GT_sbm_id']),
            'GT_degrees_sbm': np.array(gt_raw['GT_degrees_sbm']),
            'GT_degrees_spatial': np.array(gt_raw['GT_degrees_spatial'])
                }
        """
        GT = {'GT_pos': gt_pos_array}
                
        if 'P_matrix_JSON' in G.graph:
            print("P_matrix trouvée.")
            GT['GT_proba'] = np.array(json.loads(G.graph['P_matrix_JSON']))
    else:
        print("[WARNING] Aucune GroundTruth_JSON trouvée dans G.graph")
        GT = None

    if GT is not None and 'GT_pos' in GT:
            for i, node_id in enumerate(G.nodes()):
                G.nodes[node_id]['GT_pos'] = GT['GT_pos'][i]
    
    G_kept, G_hidden = hide_graph_links(G, test_size=0.10)
    G_train, G_test = hide_graph_links(G_kept, test_size=0.15)

    if spatial_ref == "deepwalk": 
        G_train = computeDistanceFeatures(G_train)
        G_kept = computeDistanceFeatures(G_kept)
        
    G_train_with_communities = computeCommunityFeatures(G_train, spatial_ref=spatial_ref)
    G_kept_with_communities = computeCommunityFeatures(G_kept, spatial_ref=spatial_ref)

    print("Save : graph pour exploration commus")
    loadsave_data_joblib(data=G_kept_with_communities, filename=f"G_kept_w_struct_com_dist_{G_name}", mode="save")
    
    
    dataset_train = prepare_balanced_data(G_test, G_train_with_communities,  negative_ratio=10.0, GroundTruth=GT)
    dataset_hidden = prepare_balanced_data(G_hidden, G_kept_with_communities, negative_ratio=50.0, GroundTruth=GT)

    print("Vérif : colonnes du dataset :")
    print(dataset_train.columns)

    print("Sauvegarde des datasets")
    save_dataset(dataset=dataset_train, filename=f"dataset_train_{G_name}")
    save_dataset(dataset=dataset_hidden, filename=f"dataset_hidden_{G_name}")
    

def analyze_commus_greels(G_name_short, nb_iterations, spatial_ref = "GT_pos", i_min =0.00, i_max = 1.00, nb_i=11, name_export_results="DATE"):
    features_GT_proba = ['GT_proba']
    features_GT_pos = ['GT_pos_dist', 
                       #'GT_spatial_deg_product', 
                    #'GT_spatial_gravity_log', 'GT_degrees_spatial_u','GT_degrees_spatial_v'
                    ]
    features_commu_inferee_normal = ["louvain_density"]
    features_commu_inferee_spatial_based_manual_iter = ["spatial_louvain_density"]
    features_commu_inferee_spatial_based_manual_iter_old = ["spatial_louvain_old_density"]
    features_commu_inferee_spatial_based_manual_iter_0_20 = ["spatial_louvain_manualiter_0_20_density"]
    features_commu_inferee_spatial_based_manual_iter_0_50 = ["spatial_louvain_manualiter_0_50_density"]
    features_commu_inferee_spatial_based_manual_iter_0_80 = ["spatial_louvain_manualiter_0_80_density"]
    features_commu_inferee_spatial_based_manual_iter_0_90 = ["spatial_louvain_manualiter_0_90_density"]
    features_commu_inferee_spatial_based_manual_reg = ["spatial_louvain_manualreg_density"]
    features_commu_inferee_spatial_based_scgravity = ["spatial_louvain_scgravity_density"]
    features_commu_inferee_spatial_based_wrdb = ["spatial_louvain_wrdb_density"]
    features_deepwalk = ["deepwalk_dist"]


    experiments = {
        "Inferred_Commu_normal": features_commu_inferee_normal,
        "Inferred_Commu_spatial_manuel_iter": features_commu_inferee_spatial_based_manual_iter,
        "Inferred_Commu_spatial_manuel_iter_old": features_commu_inferee_spatial_based_manual_iter_old,
        #"Inferred_Commu_spatial_manuel_iter_0_20": features_commu_inferee_spatial_based_manual_iter_0_20,
        #"Inferred_Commu_spatial_manuel_iter_0_50": features_commu_inferee_spatial_based_manual_iter_0_50,
        #"Inferred_Commu_spatial_manuel_iter_0_80": features_commu_inferee_spatial_based_manual_iter_0_80,
        #"Inferred_Commu_spatial_manuel_iter_0_90": features_commu_inferee_spatial_based_manual_iter_0_90,
        #"Inferred_Commu_spatial_manuel_reg": features_commu_inferee_spatial_based_manual_reg,
        #"Inferred_Commu_spatial_scgravity": features_commu_inferee_spatial_based_scgravity,
        #"Inferred_Commu_spatial_wrdb": features_commu_inferee_spatial_based_wrdb,
        #"GT_proba": features_GT_proba,
        "GT_pos": features_GT_pos,
        "GT_pos + Inferred_Commu normal": features_GT_pos + features_commu_inferee_normal,
        "GT_pos + Inferred_Commu spatial manuel iter": features_GT_pos + features_commu_inferee_spatial_based_manual_iter,
        "GT_pos + Inferred_Commu spatial manuel iter old": features_GT_pos + features_commu_inferee_spatial_based_manual_iter_old,
        #"GT_pos + Inferred_Commu spatial manuel iter 0_20": features_GT_pos + features_commu_inferee_spatial_based_manual_iter_0_20,
        #"GT_pos + Inferred_Commu spatial manuel iter 0_50": features_GT_pos + features_commu_inferee_spatial_based_manual_iter_0_50,
        #"GT_pos + Inferred_Commu spatial manuel iter 0_80": features_GT_pos + features_commu_inferee_spatial_based_manual_iter_0_80,
        #"GT_pos + Inferred_Commu spatial manuel iter 0_90": features_GT_pos + features_commu_inferee_spatial_based_manual_iter_0_90,        
    }


    all_results = []
    
    G_names_list = [
        "Airports_test",
        #"eu_airlines",
        #"faa_routes",
        #"urban_streets_savannah",
        #"urban_streets_seoul",
        #"urban_streets_washington",
        #"facebook_organizations_S1",
        #"facebook_organizations_S2",
        #"fullerene_structures_C1500"
    ]

    tasks = [f"{name}_{i}" for i in range(1) for name in G_names_list]

    cores_to_use = max(1, os.cpu_count() -2)

    print(f"Lancement de la parallélisation sur {cores_to_use} coeurs pour {len(tasks)} tâches...")

    # Exécution parallèle
    results_nested = Parallel(n_jobs=cores_to_use)(
        delayed(run_single_experiment_greels)(0, 0, "_pos", G_name_short, experiments) 
        for G_name_short in tasks
    )

    # Aplatir la liste de listes
    all_results = [item for sublist in results_nested for item in sublist]

    all_results = pd.DataFrame(all_results)
    
    output_dir = "outputs/results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"comparaison_perfs_commus_{G_name_short}_{nb_iterations}iter_{name_export_results}.csv")
    all_results.to_csv(output_path, index=False)
    print(f" Succès ! Fichier sauvegardé dans : {output_path}")
    return all_results

def run_single_experiment_greels(nb_iter, i, spatial_ref, G_name_short, experiments):
        """
        Fonction exécutée par un cœur unique pour une valeur de i et une itération donnée.
        """
        sbm_val = f"{i:.2f}"
        pos_val = f"{1-i:.2f}"
        if spatial_ref == "GT_Pos" : 
            spatial_ref = ""
        else :
            spatial_ref = f"_{spatial_ref}"
        #G_name = f"{G_name_short}_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}_{nb_iter}{spatial_ref}"
        G_name = f"{G_name_short}"
    

        # 1. Chargement des données d'entraînement
        _, dataset_train, dataset_eval, _, _, _ = load_all_data_for_graph(G_name)
        local_results = []

        for exp_name, feat_list in experiments.items():
            missing = set(feat_list) - set(dataset_train.columns)
            if missing:
                print(f" Exp {exp_name} : colonnes manquantes {missing}. Skip.")
                print(set(dataset_train.columns))
                continue

            print(f" Running: {exp_name} for SBM={i}")
        
            Params = {
                'max_depth': 3,             # Faible profondeur pour éviter l'overfitting sur 2 variables
                'learning_rate': 0.1,       # Compromis idéal vitesse/précision
                'n_estimators': 1000,       # On met beaucoup, l'early stopping fera le reste
                'subsample': 1.0,           # On garde 100% des lignes (plus stable pour peu de features)
                'colsample_bytree': 1.0,    # On garde les 2 features à chaque split
                'objective': 'binary:logistic', 
                'tree_method': 'hist',      # Accélère l'entraînement sur de gros datasets
                'reg_lambda': 1,            # Régularisation L2 pour stabiliser les poids
                'n_jobs': 1                # Utilise 1 seul coeur, pour la parallélisation
            }

            stats_df, model, _, _, _, _ = train_and_test_xgboost(dataset_train, features=feat_list, parameters = Params, plot=False)

            importances = model.feature_importances_
            feat_imp_series = pd.Series(importances, index=feat_list).sort_values(ascending=False)
                
            # Évaluation sur le dataset de référence FIXE (Graphe SBM 1.0)
            X_eval_fixed = dataset_eval[feat_list] 
            stats_eval_df = get_performance_metrics(model, X_eval_fixed, dataset_eval["target"], "EXP_")
            
            local_results.append({
                "G_name" : G_name,
                "Ratio_SBM": i,
                "Iter": nb_iter,
                "Experiment": exp_name,
                "AP_train": stats_df["Test_AP"].iloc[0],
                "AUC-ROC_train": stats_df["Test_AUC-ROC"].iloc[0],
                "AP_eval": stats_eval_df["EXP_AP"].iloc[0],
                "AUC-ROC_eval": stats_eval_df["EXP_AUC-ROC"].iloc[0],
                "Top_Feature": feat_imp_series.index[0], # On stocke la #1 pour analyse
                "Top_Importance": feat_imp_series.iloc[0]
            })

        return local_results