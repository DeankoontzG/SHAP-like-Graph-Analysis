from .utils import *
from .models import *

def execute(G, G_name) : 
    validate_input_graph(G)

    print("Validation du Graphe terminée. Lancement des calculs...")
    
    dataset, G_train = prepare_balanced_data_unknown_pos_and_community(G)

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
        "Groupe_Attributs": ['community_u', 'community_v', 'same_community', 'infomap_u', 'infomap_v', 'same_infomap'],
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
