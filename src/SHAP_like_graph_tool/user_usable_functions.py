from .utils import *
from .models import *

def execute(G) : 
    validate_input_graph(G)

    print("Validation du Graphe terminée. Lancement des calculs...")

    dataset, G_train = prepare_balanced_data_unknown_pos_and_community(G)

    dataset_with_communities = computeCommunityFeatures(G_train, dataset)

    exclude = ['u', 'v', 'target', 'label'] 
    features = [col for col in dataset_with_communities.columns if col not in exclude]

    results, model, X_test, X_train, y_test, y_train = train_and_eval_xgboost(dataset_with_communities, features=features)

    print("\n RÉSULTATS")
    print(results.to_string(index=False))

    print("\n Shapley va ! Lu.")

    shap_explanation = analyze_with_shap(model, X_test)
    save_shap_analysis(shap_explanation)





