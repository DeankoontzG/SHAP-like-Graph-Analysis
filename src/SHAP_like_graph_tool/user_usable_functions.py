from .utils import *
from .models import *

def execute(G, G_name) : 
    validate_input_graph(G)

    print("Validation du Graphe terminée. Lancement des calculs...")

    dataset, G_train = prepare_balanced_data_unknown_pos_and_community(G)

    dataset_with_communities = computeCommunityFeatures(G_train, dataset)

    dataset_with_distances = computeDistanceFeatures(G_train, dataset_with_communities)

    save_dataset(dataset_with_distances,"Airport_w_com_and_dist")

    exclude = ['u', 'v', 'target', 'label'] 
    features = [col for col in dataset_with_distances.columns if col not in exclude]

    results, model, X_test, X_train, y_test, y_train = train_and_eval_xgboost(dataset_with_communities, features=features)

    print("\n RÉSULTATS")
    print(results.to_string(index=False))

    print("\n Shapley va ! Lu.")

    shap_explanation = analyze_with_shap(model, X_test)
    save_shap_analysis(shap_explanation, filename = f"shap_explainer_{G_name}.joblib")
    

def evaluate(shap_expl):
    shap_exp = load_shap_analysis(shap_expl)

    print(shap_exp.feature_names)

    groupes = {
        "Groupe_Structure": ['cn', 'aa', 'jc', 'pa', 'sp', 'pr_u', 'pr_v', 'lcc_u', 'lcc_v', 'and_u', 'and_v', 'dc_u', 'dc_v'],
        "Groupe_Attributs": ['community_u', 'community_v', 'same_community', 'infomap_u', 'infomap_v', 'same_infomap'],
        "Groupe_Embeddings": ['n2v_p2_q0.5_cosine', 'n2v_p2_q0.5_dist', 'n2v_p1_q1_cosine', 'n2v_p1_q1_dist']
    }

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

    contradiction_ratio = (new_shap_values_abs - np.abs(new_shap_values))/(new_shap_values_abs + 1e-10)
    mean_contradiction_ratios = np.mean(contradiction_ratio, axis=0)

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

    shap.plots.bar(new_exp)
    shap.plots.bar(new_exp_abs)

    shap.plots.bar(shap_exp)




    # --- AFFICHAGE ---
    print(f"\n{'Catégorie':<25} | {'Contradiction Ratio':<20}")
    print("-" * 50)

    # On itère sur les clés du dictionnaire pour avoir les noms
    for i, cat_name in enumerate(groupes.keys()):
        print(f"{cat_name:<25} | {mean_contradiction_ratios[i]:.2}")
        print()
        





