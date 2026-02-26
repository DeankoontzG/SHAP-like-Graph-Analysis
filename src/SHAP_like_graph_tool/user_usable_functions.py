from .utils import *
from .models import *
import matplotlib.pyplot as plt
import numpy as np

def execute(G, G_name) : 
    validate_input_graph(G)

    print("Validation du Graphe terminée. Lancement des calculs...")
    
    dataset, G_train = prepare_balanced_data_unknown_pos_and_community(G, test_size = 0.15, negative_ratio=10.0)

    dataset_with_communities = computeCommunityFeatures(G_train, dataset)

    dataset_with_distances = computeDistanceFeatures(G_train, dataset_with_communities)

    save_dataset(dataset_with_distances,f"dataset_w_com_and_dist_{G_name}")

    exclude = ['u', 'v', 'target', 'label'] 
    features = [col for col in dataset_with_distances.columns if col not in exclude]

    results, model, X_test, X_train, y_test, y_train = train_and_eval_xgboost(dataset_with_communities, features=features)

    data_to_save = {
        "results": results,
        "model": model,
        "X_test": X_test,
        "X_train": X_train,
        "y_test": y_test,
        "y_train": y_train
    }

    print("Sauvegarde des données XGBoost (model,X1yTest et Train)")
    loadsave_data_joblib(data=data_to_save,filename=f"xgboost_data_{G_name}.joblib", mode="save")

    print("\n RÉSULTATS")
    print(results.to_string(index=False))

    print("\n Shapley va ! Lu.")

    shap_explanation = analyze_with_shap(model, X_test)
    print("Sauvegarde de l'analyse SHAP")
    loadsave_data_joblib(data=shap_explanation, filename = f"shap_explainer_{G_name}.joblib", mode="save")
    
def evaluate(G_name, display=False):
    shap_exp = loadsave_data_joblib(data=None, filename=f"shap_explainer_{G_name}.joblib", mode="load")

    groupes = {
        "Groupe_Structure": ['cn', 'aa', 'jc', 'pa', 'sp', 'pr_u', 'pr_v', 'lcc_u', 'lcc_v', 'and_u', 'and_v', 'dc_u', 'dc_v'],
        #"Groupe_Communities": ['community_u', 'community_v', 'same_community', 'infomap_u', 'infomap_v', 'same_infomap'],
        "Groupe_Communities": ['group_u', 'group_v',  'same_group '],
        "Groupe_Embeddings": ['n2v_p2_q0.5_cosine', 'n2v_p2_q0.5_dist', 'n2v_p1_q1_cosine', 'n2v_p1_q1_dist']
    }

    # --- CALCULS ---
    group_indices = [
        [shap_exp.feature_names.index(f) for f in features_du_groupe]
        for features_du_groupe in groupes.values()
    ]

    new_shap_values = np.array([
        shap_exp.values[:, indices].sum(axis=1) for indices in group_indices
    ]).T

    new_shap_values_abs = np.array([
        np.abs(shap_exp.values[:, indices]).sum(axis=1) for indices in group_indices
    ]).T

    # Métriques de contradiction
    contradiction_ratio = (new_shap_values_abs - np.abs(new_shap_values))/(new_shap_values_abs + 1e-10)
    mean_contradiction_ratios = np.mean(contradiction_ratio, axis=0)

    # Création des objets d'explication
    new_exp = shap.Explanation(
        values=new_shap_values,
        base_values=shap_exp.base_values,
        data=None, 
        feature_names=list(groupes.keys())
    )

    new_exp_abs = shap.Explanation(
        values=new_shap_values_abs,
        base_values=shap_exp.base_values,
        data=None, 
        feature_names=list(groupes.keys())
    )

    # Analyse Custom
    xgboost_data = loadsave_data_joblib(data=None, filename=f"xgboost_data_{G_name}.joblib", mode="load")
    shap_custom = analyse_with_shap_custom(
        model=xgboost_data["model"], 
        X_test=xgboost_data["X_test"], 
        X_train=xgboost_data["X_train"]
    )
    
    group_names = list(shap_custom.columns)
    exp_groups = shap.Explanation(
        values=shap_custom.values,
        base_values=np.array([0.5] * len(shap_custom)), 
        data=None, 
        feature_names=group_names
    )

    # --- SAUVEGARDE ---
    # On package tout ce qui est utile dans un dictionnaire
    results_to_save = {
        "G_name": G_name,
        "groupes": groupes,
        "mean_contradiction_ratios": mean_contradiction_ratios,
        "shap_explanation_grouped": new_exp,
        "shap_explanation_abs": new_exp_abs,
        "shap_custom": shap_custom,
        "exp_groups_custom": exp_groups
    }
    
    loadsave_data_joblib(
        data=results_to_save, 
        filename=f"shap_analysis_{G_name}.joblib", 
        mode="save"
    )

    # --- AFFICHAGE CONDITIONNEL ---
    if display:
        print(f"\nEvaluation avec SHAP par groupe pour {G_name}:")
        
        print("\n--- SHAP Bar Plot (Signed) ---")
        shap.plots.bar(new_exp)
        
        print("\n--- SHAP Bar Plot (Absolute) ---")
        shap.plots.bar(new_exp_abs)
        
        print("\n--- SHAP Custom Analysis ---")
        shap.plots.bar(exp_groups)

        print(f"\n{'Catégorie':<25} | {'Contradiction Ratio':<20}")
        print("-" * 50)
        for i, cat_name in enumerate(groupes.keys()):
            print(f"{cat_name:<25} | {mean_contradiction_ratios[i]:.4f}")
            
    return results_to_save

def plot_shap_evolution():

    ratios = [round(r, 2) for r in np.linspace(0, 1, 21)]
    valid_ratios = []

    all_results = {
        "base": {"Groupe_Structure": [], "Groupe_Communities": [], "Groupe_Embeddings": []},
        "abs": {"Groupe_Structure": [], "Groupe_Communities": [], "Groupe_Embeddings": []},
        "custom": {"Groupe_Structure": [], "Groupe_Communities": [], "Groupe_Embeddings": []}
    }

    for r in ratios:
        G_name = f"artificial_graph_sbm_{r:.2f}_pos_{1-r:.2f}".replace('.', '_')
        filename = f"shap_analysis_{G_name}.joblib"
        try:
            data = loadsave_data_joblib(data=None, filename=filename, mode="load")
            
            shaps = {
                "base": data["shap_explanation_grouped"],
                "abs": data["shap_explanation_abs"],
                "custom": data["exp_groups_custom"] 
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