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


###############################################################
## CONTANTES, DONT MAPPING VERS ALGOS DE CALCUL DE MÉTRIQUES ##
###############################################################
CURRENT_FILE_PATH = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE_PATH)))

EMBEDDINGS = ['n2v_homophily', 'deepwalk']
COMMUNITY_ALGOS = ['louvain', 'infomap', 'sbm']
METRICS_NODE = ["pr", "lcc", "and", "dc"]

#################################################
# FONCTIONS DE VALIDATION DES DONNES EN ENTREE ##
#################################################

def validate_input_graph(G, min_nodes=2, min_edges=1, require_undirected=True):
    """
    Vérifie la validité du graphe d'entrée avant les calculs de prédiction de liens.
    """
    # 1. Vérification du type de base
    if not isinstance(G, nx.Graph):
        raise TypeError(
            f"L'entrée doit être un objet networkx.Graph. Reçu: {type(G)}. "
            "Pour d'autres formats, convertissez-les d'abord avec networkx."
        )

    if require_undirected and G.is_directed():
        raise ValueError(
            "Le graphe est dirigé (DiGraph). L'algorithme actuel supporte uniquement "
            "les graphes non-dirigés pour garantir la validité des métriques topo."
        )

    # 3. Vérification de la taille
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    if n_nodes < min_nodes:
        raise ValueError(f"Graphe trop petit: {n_nodes} nœuds (minimum requis: {min_nodes}).")

    if n_edges < min_edges:
        raise ValueError(f"Le graphe n'a pas assez de liens ({n_edges}) pour l'entraînement.")

    # 4. Vérification optionnelle : Self-loops (peuvent fausser SP et CN)
    n_self_loops = nx.number_of_selfloops(G)
    if n_self_loops > 0:
        print(f"Warning: {n_self_loops} boucles sur soi (self-loops) détectées. "
              "Il est recommandé de les supprimer avec G.remove_edges_from(nx.selfloop_edges(G)).")

    return True


########################################
# FONCTIONS DE CALCUL DES FEATURES #####
########################################
def hide_graph_links(G, test_size = 0.15):
    all_edges = list(G.edges())
    random.seed(42)
    random.shuffle(all_edges)
    
    split_idx = int(len(all_edges) * (1 - test_size))
    train_edges = all_edges[:split_idx]
    test_edges = all_edges[split_idx:]
    
    # 2. Création du graphe d'entraînement (G sans le test set)
    # C'est sur ce graphe qu'on va tout calculer
    G_train = nx.Graph()
    G_train.add_nodes_from(G.nodes())
    G_train.add_edges_from(train_edges)

    G_eval = nx.Graph()
    G_eval.add_nodes_from(G.nodes())
    G_eval.add_edges_from(test_edges)
    
    print(f"Graphe original: {G.number_of_edges()} liens")
    print(f"Graphe d'entraînement: {G_train.number_of_edges()} liens")
    print(f"Liens cachés pour le test: {len(test_edges)}")

    return G_train, G_eval


def _extract_pair_features(G_train, u, v, densities):
    """
    Aggrège les infos de noeuds (IDs de blocs, centralités) et 
    calcule les métriques de paires à la volée.
    """
    nu = G_train.nodes[u]
    nv = G_train.nodes[v]

    # On retire le lien si il existe, pour ne pas polluer les heuristiques structurelles. 
    #Impact mathématique direct sur Jaccard, PA et SP, indirect sur AA, sur CN je suis pas sûr
    has_edge = G_train.has_edge(u, v)
    if has_edge:
        G_train.remove_edge(u, v)   

    features = {
        'cn': len(list(nx.common_neighbors(G_train, u, v))),
        'aa': next(nx.adamic_adar_index(G_train, [(u, v)]))[2],
        'jc': next(nx.jaccard_coefficient(G_train, [(u, v)]))[2],
        'pa': next(nx.preferential_attachment(G_train, [(u, v)]))[2],
        'sp': nx.shortest_path_length(G_train, u, v) if nx.has_path(G_train, u, v) else 0
    }

    if has_edge:
        G_train.add_edge(u, v)

    for metric in METRICS_NODE:
        features[f'{metric}_u'] = nu.get(metric, 0)
        features[f'{metric}_v'] = nv.get(metric, 0)


    for algo in COMMUNITY_ALGOS:
        id_u = nu.get(f'{algo}_id')
        id_v = nv.get(f'{algo}_id')
        pair = tuple(sorted((id_u, id_v)))

        features[f'{algo}_density'] = densities[algo].get(pair, 0)
        features[f'same_{algo}'] = 1 if id_u == id_v else 0

    for emb in EMBEDDINGS:
        if emb in nu and emb in nv:
            vec_u = nu[emb].reshape(1, -1)
            vec_v = nv[emb].reshape(1, -1)
            features[f'{emb}_cos'] = cosine_similarity(vec_u, vec_v)[0][0]
            features[f'{emb}_dist'] = np.linalg.norm(vec_u - vec_v)

    return features

def prepare_balanced_data(G, G_train, negative_ratio=10.0, seed=42):
    """
    Prépare le dataset final en utilisant G_train pour les features
    et G pour vérifier l'existence réelle des liens (target).
    """
    random.seed(seed)
    all_edges = list(G.edges())
    nodes = list(G.nodes())
    n_pos = len(all_edges)
    data = []
    densities = prepare_all_densities(G_train) # Calcul des densités inter blocs pour les commu

    print(f"Extraction des features pour {n_pos} liens positifs...")
    # --- 1. CLASSE POSITIVE ---
    for u, v in all_edges:
        features = _extract_pair_features(G_train, u, v, densities)
        row = {'u': u, 'v': v, 'target': 1}
        row.update(features)
        data.append(row)
    
    # --- 2. CLASSE NÉGATIVE ---
    n_neg_target = int(n_pos * negative_ratio)
    print(f"Génération de {n_neg_target} non-liens (ratio {negative_ratio})...")
    
    neg_count = 0
    while neg_count < n_neg_target:
        u, v = random.sample(nodes, 2)
        if u != v and not G.has_edge(u, v):
            features = _extract_pair_features(G_train, u, v, densities)
            row = {'u': u, 'v': v, 'target': 0}
            row.update(features)
            data.append(row)
            neg_count += 1

    df = pd.DataFrame(data)
    print(f"DataFrame créé avec succès : {df.shape[0]} lignes.")
    return df

### Fonction de calcul de features de structure des noeuds
def computeStructureFeatures(G_train):
    print("\n--- Enrichissement du Graphe avec les Métriques de Structure ---")
    print("Calcul : PageRank, Clustering, Average Neighbor Degree, Degree Centrality")
    pr = nx.pagerank(G_train)
    lcc = nx.clustering(G_train)
    avg_nd = nx.average_neighbor_degree(G_train)
    dc = nx.degree_centrality(G_train)

    for node in G_train.nodes():
        G_train.nodes[node].update({
            'pr': pr.get(node, 0),
            'lcc': lcc.get(node, 0),
            'and': avg_nd.get(node, 0),
            'dc': dc.get(node, 0)
        })
    
    return G_train

### Fonction parente qui appelle les différentes fonctions de calcul de features de communauté

def _appendLouvainCommunities(G_train):
    communities = nx.community.louvain_communities(G_train, seed=42)

    node_to_community = {} 
    for i, community in enumerate(communities):
        for node in community:
            node_to_community[node] = i

    nx.set_node_attributes(G_train, node_to_community, "louvain_id")
    _normalize_community_assignment(G_train, "louvain_id")

def _appendInfomapCommunities(G_train):
    im = Infomap("--two-level --silent")
    
    nodes_list = list(G_train.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes_list)}
    idx_to_node = {i: node for i, node in enumerate(nodes_list)}
    
    for source, target in G_train.edges():
        im.add_link(node_to_idx[source], node_to_idx[target])
    
    im.run()

    node_to_infomap = {}
    for node in im.tree:
        if node.is_leaf:
            original_node_name = idx_to_node[node.node_id]
            node_to_infomap[original_node_name] = node.module_id

    nx.set_node_attributes(G_train, node_to_infomap, "infomap_id")
    _normalize_community_assignment(G_train, "infomap_id")


def _appendGraphToolSBM(G_train):
    """
    Inférence SBM via graph-tool avec détection automatique 
    du nombre de blocs (MDL).
    """
    nodes_list = list(G_train.nodes())
    node_index = {node: i for i, node in enumerate(nodes_list)}
    
    G_gt = gt.Graph(directed=False)
    G_gt.add_vertex(len(nodes_list))
    
    edges = [(node_index[u], node_index[v]) for u, v in G_train.edges()]
    G_gt.add_edge_list(edges)

    state = gt.minimize_blockmodel_dl(G_gt)
    blocks = state.get_blocks()
    
    node_to_community = {nodes_list[i]: int(blocks[i]) for i in range(len(nodes_list))}
            
    nx.set_node_attributes(G_train, node_to_community, "sbm_id")
    _normalize_community_assignment(G_train, "sbm_id")

def _normalize_community_assignment(G, attr_name):
    """ Remplace les NaN par des IDs uniques (singletons) """
    nodes_data = nx.get_node_attributes(G, attr_name)
    
    current_ids = [int(v) for v in nodes_data.values() if pd.notnull(v)]
    next_id = max(current_ids) + 1 if current_ids else 0
    
    mapping = {}
    for node in G.nodes():
        val = nodes_data.get(node)
        if pd.isnull(val):
            mapping[node] = next_id
            next_id += 1
        else:
            mapping[node] = val
            
    nx.set_node_attributes(G, mapping, attr_name)

COMMUNITY_MAPPING = {
    'louvain': _appendLouvainCommunities,
    'infomap': _appendInfomapCommunities,
    'sbm': _appendGraphToolSBM
}

def computeCommunityFeatures(G_train, algos="All"):
    print("\n--- Enrichissement du Graphe avec les Communautés ---")
    to_run = COMMUNITY_ALGOS if algos == "All" else algos
    
    for algo in to_run:
        if algo in COMMUNITY_MAPPING:
            print(f"Calcul des communautés via {algo}...")
            COMMUNITY_MAPPING[algo](G_train)
        else:
            print(f"Attention : L'algorithme {algo} n'est pas reconnu.")
            
    return G_train

def prepare_all_densities(G_train):
    """
    Pré-calcule les densités de blocs pour tous les algorithmes.
    """
    all_densities = {}
    
    for algo in COMMUNITY_ALGOS:
        attr_name = f"{algo}_id"
        node_to_block = nx.get_node_attributes(G_train, attr_name)
        
        # 1. Compter les membres par bloc
        block_sizes = pd.Series(node_to_block).value_counts().to_dict()
        blocks = list(block_sizes.keys())
        
        # 2. Compter les liens réels entre blocs (uniquement sur le triangle supérieur)
        counts = {(b1, b2): 0 for i, b1 in enumerate(blocks) for b2 in blocks[i:]}
        
        for u, v in G_train.edges():
            bu, bv = node_to_block.get(u), node_to_block.get(v)
            if bu is not None and bv is not None:
                pair = tuple(sorted((bu, bv)))
                if pair in counts:
                    counts[pair] += 1
        
        # 3. Calculer les densités (Lien Réels / Liens Possibles)
        algo_densities = {}
        for (b1, b2), real_count in counts.items():
            n1, n2 = block_sizes[b1], block_sizes[b2]
            if b1 == b2:
                possible = (n1 * (n1 - 1)) / 2  # Combinaisons intra
            else:
                possible = n1 * n2              # Combinaisons inter
            
            algo_densities[(b1, b2)] = real_count / possible if possible > 0 else 0
            
        all_densities[algo] = algo_densities
        
    return all_densities

### Fonction parente qui appelle les différentes fonctions de calcul de features de distance

def _append_node2vec_features(G_train, p, q, attr_name,dimensions=64):
    """
    Génère les embeddings Node2Vec et retourne un dictionnaire {node_id: vector}
    """
    print(f"Calcul de Node2Vec (p={p}, q={q})...")
    print(f"Génération des marches aléatoires (dim={dimensions})...")
    
    # Configuration de Node2Vec
    # p=1, q=1 -> équivalent à DeepWalk
    # p=1, q=2 -> Favorise l'exploration locale (structure)
    # p=2, q=0.5 -> Favorise l'exploration lointaine (communautés) - homophilie
    node2vec = Node2Vec(G_train, 
                        dimensions=dimensions, 
                        walk_length=30, 
                        num_walks=100, 
                        workers=4, 
                        p=p, q=q)

    print("Entraînement du modèle Skip-gram...")
    try:
        model = node2vec.fit(window=10, min_count=1, batch_words=4, vector_size=dimensions)
    except TypeError:
        model = node2vec.fit(window=10, min_count=1, batch_words=4, size=dimensions)
    
    # On récupère les vecteurs dans un dictionnaire
    embeddings = {node: model.wv[str(node)] for node in G_train.nodes()}
    nx.set_node_attributes(G_train, embeddings, attr_name)

EMBEDDING_MAPPING = {
    'n2v_homophily': lambda G: _append_node2vec_features(G, p=2, q=0.5, attr_name="n2v_homophily"),
    'deepwalk': lambda G: _append_node2vec_features(G, p=1, q=1, attr_name="deepwalk")
}

def apply_fixed_log_binning(df, col_name, num_bins=10):
    """
    Découpe en bins par ordre de grandeur (log10 appliqué sur valeurs, puis découpe linéaire sur valeur)
    """
    epsilon = 1e-9
    vals_log = np.log10(df[col_name] + epsilon)
    
    bins = np.linspace(vals_log.min(), vals_log.max(), num_bins + 1)
    
    df[f'{col_name}_log_bin'] = pd.cut(
        vals_log, 
        bins=bins, 
        labels=False, 
        include_lowest=True
    )
    return df

def apply_quantile_binning(df, col_name, num_bins=10):
    """
    Découpe en bins contenant le même nombre d'observations (10% par bin).
    """
    df[f'{col_name}_quantile_bin'] = pd.qcut(
        df[col_name], 
        q=num_bins, 
        labels=False, 
        duplicates='drop'
    )
    return df

def computeDistanceFeatures(G_train, embeddings="All"):
    to_run = EMBEDDINGS if embeddings == "All" else embeddings
    print("\n--- Enrichissement du Graphe avec les Embeddings ---")

    for emb in to_run:
        if emb in EMBEDDING_MAPPING:
            print(f"Calcul des embeddings via {emb}...")
            EMBEDDING_MAPPING[emb](G_train)
        else:
            print(f"Attention : L'algorithme {emb} n'est pas reconnu.")
    return G_train

def enrich_dataset_with_ground_truth(df, G, p_intra = 7517986, q_inter = 0.0002 ):
    """
    Ajoute les infos réelles de position et block du graphe initial, pour les graphes générés artificiellement.
    """
    pos_dict = nx.get_node_attributes(G, 'pos')
    block_dict = nx.get_node_attributes(G, 'block')

    df['block_reel_u'] = df['u'].map(block_dict)
    df['block_reel_v'] = df['v'].map(block_dict)
    df['same_block_reel'] = (df['block_reel_u'] == df['block_reel_v']).astype(int)

    df['proba_lien_reelle'] = np.where(df['same_block_reel'] == 1, p_intra, q_inter)


    def calculate_dist(row):
        u, v = row['u'], row['v']
        if u in pos_dict and v in pos_dict:
            p1 = np.array(pos_dict[u])
            p2 = np.array(pos_dict[v])
            return np.linalg.norm(p1 - p2)
        return None

    print("Calcul des distances réelles...")
    df['dist_reelle'] = df.apply(calculate_dist, axis=1)

    return df

#################################################
######### FONCTIONS DE CROSS VALIDATION #########
#################################################

def k_fold_cross_validation(G_train, k=2, features_list=None, n_trials=50, graph_name="G_NAME"):
    folds_data = _prepare_precalculated_folds(G_train, k=k)
    study = _run_optuna_tuning(folds_data, features_list, n_trials=n_trials)
    
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

def _prepare_precalculated_folds(G_train, k=1):
    edges = list(G_train.edges())
    if k == 1:
        folds_idx = [train_test_split(range(len(edges)), test_size=0.2)]
    else:
        kf = KFold(n_splits=k, shuffle=True)
        folds_idx = list(kf.split(edges))

    precalculated_folds = []

    for f_idx, (train_idx, val_idx) in enumerate(folds_idx):
        print(f"--- Pré-calcul Fold {f_idx + 1} ---")
        train_edges = [edges[i] for i in train_idx]
        current_val_edges = [edges[i] for i in val_idx]
        
        G_fold_train = nx.Graph()
        G_fold_train.add_nodes_from(G_train.nodes(data=True))
        G_fold_train.add_edges_from(train_edges)

        G_fold_val = nx.Graph()
        G_fold_val.add_nodes_from(G_train.nodes(data=True))
        G_fold_val.add_edges_from(current_val_edges)
        
        G_fold_train = computeStructureFeatures(G_fold_train)
        G_fold_train = computeCommunityFeatures(G_fold_train)
        G_fold_train = computeDistanceFeatures(G_fold_train)

        ds_train = prepare_balanced_data(G_fold_train, G_fold_train, negative_ratio=10.0)
        ds_val = prepare_balanced_data(G_fold_val, G_fold_train, negative_ratio=25.0)
        
        precalculated_folds.append((ds_train, ds_val))
        
    return precalculated_folds

def _run_optuna_tuning(precalculated_folds, features_list=None, n_trials=50):

    if features_list is None or len(features_list) == 0:
        exclude = ['u', 'v', 'target', 'label']
        features = [col for col in precalculated_folds[0][0].columns if col not in exclude]
        print(f"Features détectées ({len(features)}) : {features}")
    else:
        features = features_list

    def objective(trial):
        params = {
            'n_estimators': 150,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'tree_method': 'hist',
            'random_state': 42
        }

        f_auc_v, f_auc_t, f_ap_v = [], [], []

        for ds_train, ds_val in precalculated_folds:
            model = XGBClassifier(**params)
            model.fit(ds_train[features], ds_train['target'])
            
            p_val = model.predict_proba(ds_val[features])[:, 1]
            p_train = model.predict_proba(ds_train[features])[:, 1]
            
            f_auc_v.append(roc_auc_score(ds_val['target'], p_val))
            f_auc_t.append(roc_auc_score(ds_train['target'], p_train))
            f_ap_v.append(average_precision_score(ds_val['target'], p_val))
        
        avg_auc_v = np.mean(f_auc_v)
        trial.set_user_attr("std_auc", np.std(f_auc_v))
        trial.set_user_attr("avg_ap", np.mean(f_ap_v))
        trial.set_user_attr("delta_auc", np.mean(f_auc_t) - avg_auc_v)

        return avg_auc_v

    study = optuna.create_study(direction='maximize')
    baseline = {'learning_rate': 0.1, 'max_depth': 6, 'min_child_weight': 6,
        'subsample': 1.0, 'colsample_bytree': 1.0, 'reg_alpha': 1e-3, 'reg_lambda': 1.0
    }
    study.enqueue_trial(baseline)
    study.optimize(objective, n_trials=n_trials)
    
    return study

########################################
# FONCTIONS D'APPEL DE SHAP ############
########################################

def analyze_with_shap(model, X_test, y_test, max_pos=2000, negative_ratio=1.0):
    """
    Calcule les SHAP values sur un échantillon équilibré.
    Par défaut : tous les positifs (max 2000) et autant de négatifs.
    """
    pos_indices = y_test[y_test == 1].index
    neg_indices = y_test[y_test == 0].index

    # Plafonnage de la classe positive
    n_pos = min(len(pos_indices), max_pos)
    pos_sample = pos_indices[:n_pos]

    # Échantillonnage de la classe négative (ratio 1:1 par défaut)
    n_neg = int(n_pos * negative_ratio)
    neg_sample = y_test.loc[neg_indices].sample(n=n_neg, random_state=42).index

    X_shap = X_test.loc[pos_sample.union(neg_sample)]
    
    print(f"Analyse SHAP : {len(pos_sample)} positifs et {len(neg_sample)} négatifs (Total: {len(X_shap)})")

    # Configuration de l'explainer 'Boîte Noire' (le plus stable sur mon Mac)
    # On définit la fonction de prédiction (proba de la classe 1)
    model_predict = lambda x: model.predict_proba(x)[:, 1]
    
    # Utilisation d'un masker (échantillon de référence)
    # On prend 50 lignes pour équilibrer vitesse et précision
    masker = X_test.iloc[:50]
    
    # Initialisation de l'explainer
    explainer = shap.Explainer(model_predict, masker)    
    
    # 2. Calcul effectif des SHAP values
    # On récupère l'objet 'Explanation' complet
    shap_explanation = explainer(X_shap)
    
    # 3. Extraction des valeurs numériques pour le retour de fonction
    # On récupère les valeurs brutes (.values)
    shap_values = shap_explanation.values

    # Gestion de la dimension (si SHAP renvoie [n_samples, n_features, 2])
    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]
    
    return shap_explanation

def display_shap(graphname, output_dir="outputs/plots"):

    filename = f"shap_explainer_{graphname}.joblib"
    shap_explainer = loadsave_data_joblib(data=None, filename=filename, mode="load")

    # --- GÉNÉRATION DES PLOTS ---
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Summary Points (Beeswarm)
    plt.figure(figsize=(12, 8))
    # On peut passer l'objet explanation directement, c'est plus moderne
    shap.plots.beeswarm(shap_explainer, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"shap_summary_points_{graphname}.png"))
    plt.close()

    # Plot 2: Summary Bar
    plt.figure(figsize=(12, 8))
    shap.plots.bar(shap_explainer, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"shap_summary_bar_{graphname}.png"))
    plt.close()

    return 1

def analyse_with_shap_custom(model, X_eval, X_train, baseline="general", output_dir="outputs/plots"):
    groupes = {
        "Groupe_Structure": ['cn', 'aa', 'jc', 'pa', 'sp', 'pr_u', 'pr_v', 'lcc_u', 'lcc_v', 'and_u', 'and_v', 'dc_u', 'dc_v'],
        "Groupe_Communities": ['sbm_u', 'sbm_v', 'same_sbm', 'infomap_u', 'infomap_v', 'same_infomap',"louvain_u","louvain_v","same_louvain"],
        #"Groupe_Communities": ['sbm_density', 'same_sbm', 'infomap_density', 'same_infomap',"louvain_density", "same_louvain"],
        #"Groupe_Communities": ['group_u', 'group_v',  'same_group'],
        "Groupe_Embeddings": ['n2v_homophily_cos', 'n2v_homophily_dist', 'deepwalk_cos', 'deepwalk_dist']
    }

    if baseline=="CaseByCase" : 
        print("Custom SHAP baseline : case by case")
        baseline_map = {
            # Zéro pour la structure : 
            "cn": 0, "aa": 0, "pa": 0,  
            # Moyenne pour le continu :
            "n2v_homophily_cos": X_train["n2v_homophily_cos"].mean(),
            "n2v_homophily_dist": X_train["n2v_homophily_dist"].mean(),
            "deepwalk_cos": X_train["deepwalk_cos"].mean(),
            "deepwalk_dist": X_train["deepwalk_dist"].mean(),
            # Mode pour le catégoriel
            "community_u": X_train["community_u"].mode()[0] # Mode pour le catégoriel
        }
    
    elif baseline =="general" :
        print("Custom SHAP baseline : general")
        baseline_map = {}
        # Structure -> Zéro (Absence de connexion)
        for m in groupes["Groupe_Structure"]:
            baseline_map[m] = 0
        # Attributs -> -1 (Catégorie inconnue)
        for m in groupes["Groupe_Communities"]:
            baseline_map[m] = -1
        # Embeddings -> Moyenne (Bruit blanc statistique)
        means = X_train[groupes["Groupe_Embeddings"]].mean()
        for m in groupes["Groupe_Embeddings"]:
            baseline_map[m] = means[m]

    else: 
        print("Custom SHAP baseline : Mean for all")
        baseline_map = {}
        overall_means = X_train.mean()
        for group_name, features in groupes.items():
            for f in features:
                baseline_map[f] = overall_means[f]

    n_groups = len(groupes)
    group_names = list(groupes.keys())

    def _get_val_for_coalition(coalition_mask, sample):
        """
        coalition_mask: liste de booléens (ex: [True, False, True]) 
        indiquant quels groupes on garde.
        """
        x_mapped = sample.copy()
        for i, name in enumerate(group_names):
            if not coalition_mask[i]:
                indices = groupes[name]
                # On applique la baseline spécifique à chaque métrique du groupe
                x_mapped[indices] = [baseline_map[m] for m in indices]
        return model.predict_proba([x_mapped])[0][1]

    # Stockage des SHAP values finales
    shap_values_coalition = np.zeros((len(X_eval), n_groups))

    # 3. Boucle sur chaque échantillon (Sample)
    # --- Calcul des SHAP values ---
    for idx in range(len(X_eval)):
        x_sample = X_eval.iloc[idx]
        
        for i in range(n_groups):
            phi_i = 0
            other_indices = [g for g in range(n_groups) if g != i]
            
            for r in range(len(other_indices) + 1):
                for subset in itertools.combinations(other_indices, r):
                    # Poids de Shapley
                    weight = factorial(len(subset)) * factorial(n_groups - len(subset) - 1) / factorial(n_groups)
                    
                    # Masques
                    mask_S = [False] * n_groups
                    for s_idx in subset: mask_S[s_idx] = True
                    
                    mask_Si = list(mask_S)
                    mask_Si[i] = True
                    
                    # Différence marginale
                    v_Si = _get_val_for_coalition(mask_Si, x_sample)
                    v_S = _get_val_for_coalition(mask_S, x_sample)
                    
                    phi_i += weight * (v_Si - v_S)
            
            shap_values_coalition[idx, i] = phi_i

    return pd.DataFrame(shap_values_coalition, columns=group_names)

def calculate_feature_rankings(shap_values, feature_names, top_k, plot = False, output_dir="outputs/plots",):
    """Calcule la distribution des rangs et génère le barplot du Top 5."""
    abs_shap = np.abs(shap_values)
    ranks = np.argsort(-abs_shap, axis=1)
    
    ranking_stats = {}
    n_samples, n_features = shap_values.shape

    for i, name in enumerate(feature_names):
        feature_ranks = np.where(ranks == i)[1] + 1
        counts = np.bincount(feature_ranks, minlength=n_features + 1)[1:]
        ranking_stats[name] = (counts / n_samples) * 100

    df_ranks = pd.DataFrame(ranking_stats, index=[f"Rang {i+1}" for i in range(n_features)])
    
    if plot : 
        top_k_displayed = df_ranks.iloc[0:top_k, :].sum(axis=0).sort_values(ascending=False)
        
        plt.figure(figsize=(12, 7))
        
        sns.barplot(
            x=top_k_displayed.index, 
            y=top_k_displayed.values, 
            hue=top_k_displayed.index, 
            palette="viridis", 
            legend=False
        )
        
        plt.title(f"Importance structurelle : % de présence dans le Top {top_k} SHAP")
        plt.ylabel("% de présence")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
    
    return df_ranks

########################################
## FONCTIONS UTILITAIRES DE LOAD SAVE ##
########################################

def save_dataset(dataset, filename="dataset"):
    output_dir = os.path.join(PROJECT_ROOT, "outputs", "results")
    output_path = os.path.join(output_dir, filename)
    
    # Création du dossier (absolu)
    os.makedirs(output_dir, exist_ok=True)
    
    dataset.to_parquet(output_path, index=False)
    print(f"Dataset (DataFrame) sauvegardé : {output_path}")

    return output_path

def load_dataset(filename="dataset", talk = False):
    input_dir = os.path.join(PROJECT_ROOT, "outputs", "results")
    input_path = os.path.join(input_dir, filename)
    
    if not os.path.exists(input_path) :
        print(f"Erreur : Le fichier n'existe pas : {input_path}")
        return None
    
    dataset = pd.read_parquet(input_path)
    if talk :
        print(f" Dataset chargé avec succès depuis : {input_path}")
        print(f" Taille : {dataset.shape[0]} lignes, {dataset.shape[1]} colonnes.")
    
    return dataset

def loadsave_data_joblib(data=None, filename="data.joblib", mode="save", talk=False):
    """
    Gère la sauvegarde et le chargement d'objets en .joblib (SHAP, XGBoost, etc.).
    """
    base_path = Path(PROJECT_ROOT) if 'PROJECT_ROOT' in globals() else Path.cwd()
    target_path = base_path / "outputs" / "results" / filename

    if mode == "save":
        if data is None :
            print("Erreur : Aucun objet fourni pour la sauvegarde.")
            return None
        
        # Création du dossier
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(data, target_path, compress=3)
        if talk :
            print(f"Objet sauvegardé dans : {target_path}")
        return target_path

    elif mode == "load":
        if not target_path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {target_path}")
        
        obj = joblib.load(target_path)
        if talk :
            print(f"Objet chargé avec succès depuis : {target_path}")
        
        return obj

def load_all_data_for_graph(G_name, talk=False):
    # 1. G_train (avec structure, communautés et distances)
    try:
        G_train = loadsave_data_joblib(data=None, filename=f"G_train_w_struct_com_dist_{G_name}", mode="load", talk = talk)
    except Exception:
        print(f"G_train introuvable pour {G_name}, création d'un graphe vide.")
        G_train = nx.Graph()

    # 2. Dataset de Train (via load_dataset)
    try:
        dataset_train = load_dataset(filename=f"dataset_train_{G_name}", talk = talk)
    except Exception:
        print(f"Dataset de Train introuvable pour {G_name}.")
        dataset_train = None

    # 3. Dataset d'Évaluation (via load_dataset)
    try:
        dataset_eval = load_dataset(filename=f"dataset_eval_{G_name}", talk = talk)
    except Exception:
        print(f"Dataset d'Évaluation introuvable pour {G_name}.")
        dataset_eval = None

    # 4. Données XGBoost (Modèle, X_test, etc.)
    try:
        xgboost_data = loadsave_data_joblib(data=None, filename=f"xgboost_data_{G_name}.joblib", mode="load", talk = talk)
    except Exception:
        print(f"Données XGBoost introuvables pour {G_name}.")
        xgboost_data = None

    # 5. SHAP Explainer
    try:
        shap_explainer = loadsave_data_joblib(data=None, filename=f"shap_explainer_{G_name}.joblib", mode="load", talk = talk)
    except Exception:
        print(f"SHAP Explainer introuvable pour {G_name}.")
        shap_explainer = None

    # 6. SHAP Analysis
    try:
        shap_analysis = loadsave_data_joblib(data=None, filename=f"shap_analysis_{G_name}.joblib", mode="load", talk = talk)
    except Exception:
        print(f"SHAP Analysis introuvable pour {G_name}.")
        shap_analysis = None

    return G_train, dataset_train, dataset_eval, xgboost_data, shap_explainer, shap_analysis

class GraphEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)

def save_graph(G, filename):
    base_path = Path(PROJECT_ROOT) if 'PROJECT_ROOT' in globals() else Path.cwd()
    target_path = base_path / "outputs" / "results" / filename

    data = nx.node_link_data(G)
    with open(filename, 'w') as f:
        json.dump(data, f, cls=GraphEncoder)
    print(f"Graphe sauvegardé dans {filename}")

def load_graph(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return nx.node_link_graph(data)