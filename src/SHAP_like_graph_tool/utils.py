import random
from re import X
import numpy as np
import pandas as pd
import networkx as nx
import itertools
from math import factorial
from networkx.algorithms.community import louvain_communities
from sklearn.metrics.pairwise import cosine_similarity
from node2vec import Node2Vec
from infomap import Infomap
import graph_tool.all as gt
import shap
import os
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

CURRENT_FILE_PATH = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE_PATH)))

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
def _get_topology_features(G, u, v, precomputed, is_existing_edge=False):
    """Calcule les métriques topologiques pour une paire (u, v)"""
    
    # 1. Métriques de paires (Voisinage)
    aa = next(nx.adamic_adar_index(G, [(u, v)]))[2]
    jc = next(nx.jaccard_coefficient(G, [(u, v)]))[2]
    pa = next(nx.preferential_attachment(G, [(u, v)]))[2]
    cn = len(list(nx.common_neighbors(G, u, v)))

    try:
        sp = nx.shortest_path_length(G, source=u, target=v)
    except nx.NetworkXNoPath:
        sp = 0 

    # 2. Métriques de Nœuds (extraites du dictionnaire pré-calculé)
    # On ajoute les versions pour u et pour v
    node_features = {
        'pr_u': precomputed['pr'].get(u, 0), 'pr_v': precomputed['pr'].get(v, 0),
        'lcc_u': precomputed['lcc'].get(u, 0), 'lcc_v': precomputed['lcc'].get(v, 0),
        'and_u': precomputed['and'].get(u, 0), 'and_v': precomputed['and'].get(v, 0),
        'dc_u': precomputed['dc'].get(u, 0), 'dc_v': precomputed['dc'].get(v, 0)
    }

    # Fusion de toutes les métriques
    topo_res = {'cn': cn, 'aa': aa, 'jc': jc, 'pa': pa, 
                'sp': sp
               }
    topo_res.update(node_features)
    
    return topo_res


def prepare_balanced_data_unknown_pos_and_community(G, test_size = 0.15, negative_ratio=1.0):
    all_edges = list(G.edges())
    nodes = list(G.nodes())
    n_pos = len(all_edges)
    data = []
    random.seed(42)

    # 1. Extraction des arêtes pour le split
    random.shuffle(all_edges)
    
    split_idx = int(len(all_edges) * (1 - test_size))
    train_edges = all_edges[:split_idx]
    test_edges = all_edges[split_idx:]
    
    # 2. Création du graphe d'entraînement (G sans le test set)
    # C'est sur ce graphe qu'on va tout calculer
    G_train = nx.Graph()
    G_train.add_nodes_from(G.nodes())
    G_train.add_edges_from(train_edges)
    
    print(f"Graphe original: {G.number_of_edges()} liens")
    print(f"Graphe d'entraînement: {G_train.number_of_edges()} liens")
    print(f"Liens cachés pour le test: {len(test_edges)}")
    

    # --- ÉTAPE DE PRÉ-CALCUL ---
    # On calcule les métriques de noeuds une seule fois ici
    print("Pré-calcul des métriques de nœuds...")
    precomputed = {
        'pr': nx.pagerank(G_train),                    # PageRank (PR)
        'lcc': nx.clustering(G_train),                # Local Clustering Coefficient (LCC)
        'and': nx.average_neighbor_degree(G_train),   # Average Neighbor Degree (AND)
        'dc': nx.degree_centrality(G_train)           # Degree Centrality (DC)
    }
    
    # --- 1. CLASSE POSITIVE ---
    for u, v in all_edges:
        topo = _get_topology_features(G_train, u, v, precomputed, is_existing_edge=True)
        
        row = {
            'u': u, 
            'v': v,
            'target': 1
        }
        row.update(topo)
        data.append(row)
    
    # --- 2. CLASSE NÉGATIVE ---
    n_neg_target = int(n_pos * negative_ratio)
    neg_count = 0
    while neg_count < n_neg_target:
        u, v = random.sample(nodes, 2)
        if not G.has_edge(u, v) and u != v:
            topo = _get_topology_features(G_train, u, v, precomputed, is_existing_edge=False)
            
            row = {
                'u': u, 
                'v': v,
                'target': 0
            }
            row.update(topo)
            data.append(row)
            neg_count += 1

    print(f"DataFrame créé <3 : {len(data)} paires de noeuds choisies")
    return pd.DataFrame(data), G_train

### Fonction parente qui appelle les différentes fonctions de calcul de features de communauté
def computeCommunityFeatures(G, dataFrame, features = "All"):
    print("\n")
    print("--- Calcul des métriques de communauté ---")
    print("Calcul des communautés de louvain...")
    dataFrame = _appendLouvainCommunities(G, dataFrame=dataFrame)
    print("Calcul des communautés via Infomap...")
    dataFrame = _appendInfomapCommunities(G, dataFrame=dataFrame)
    print("Calcul des communautés via SBM...")
    dataFrame = _appendGraphToolSBM(G, dataFrame=dataFrame)

    return dataFrame

def _appendLouvainCommunities(G, dataFrame):
    communities = nx.community.louvain_communities(G, seed=42)

    node_to_community = {} 
    for i, community in enumerate(communities):
        for node in community:
            node_to_community[node] = i
            
    louvain_communities_data = dataFrame.copy()

    louvain_communities_data["louvain_u"] = louvain_communities_data["u"].map(node_to_community)
    louvain_communities_data["louvain_v"] = louvain_communities_data["v"].map(node_to_community)
    louvain_communities_data["same_louvain"] = (louvain_communities_data["louvain_u"] == louvain_communities_data["louvain_v"]).astype(int)
    
    return louvain_communities_data

def _appendInfomapCommunities(G, dataFrame):

    im = Infomap("--two-level --silent")
    
    for source, target in G.edges():
        im.add_link(int(source), int(target))
    
    im.run()

    node_to_infomap = {node.node_id: node.module_id for node in im.tree if node.is_leaf}

    infomap_data = dataFrame.copy()

    infomap_data["u"] = pd.to_numeric(infomap_data["u"], errors='coerce').astype(int)
    infomap_data["v"] = pd.to_numeric(infomap_data["v"], errors='coerce').astype(int)

    infomap_data["infomap_u"] = infomap_data["u"].map(node_to_infomap)
    infomap_data["infomap_v"] = infomap_data["v"].map(node_to_infomap)
    infomap_data["same_infomap"] = (infomap_data["infomap_u"] == infomap_data["infomap_v"]).astype(int)

    return infomap_data

def _appendGraphToolSBM(G_nx, dataFrame):
    """
    Inférence SBM via graph-tool avec détection automatique 
    du nombre de blocs (MDL).
    """
    nodes_list = list(G_nx.nodes())
    node_index = {node: i for i, node in enumerate(nodes_list)}
    
    G_gt = gt.Graph(directed=False)
    G_gt.add_vertex(len(nodes_list))
    
    edges = [(node_index[u], node_index[v]) for u, v in G_nx.edges()]
    G_gt.add_edge_list(edges)

    state = gt.minimize_blockmodel_dl(G_gt)

    blocks = state.get_blocks()
    
    node_to_community = {nodes_list[i]: int(blocks[i]) for i in range(len(nodes_list))}
            
    sbm_data = dataFrame.copy()
    sbm_data["sbm_community_u"] = sbm_data["u"].map(node_to_community)
    sbm_data["sbm_community_v"] = sbm_data["v"].map(node_to_community)
    sbm_data["same_sbm_community"] = (sbm_data["sbm_community_u"] == sbm_data["sbm_community_v"]).astype(int)
    
    return sbm_data

### Fonction parente qui appelle les différentes fonctions de calcul de features de distance
def computeDistanceFeatures(G, dataFrame, features = "All"):
    print("\n")
    print("--- Calcul des métriques de distance ---")
    print("Calcul de Node2Vec homophilie p=2, q=0.5 ...")
    dataFrame = _append_node2vec_features(G, dataFrame=dataFrame, p=2, q=0.5)
    print("Calcul de DeepWalk p=1, q=1 ...")
    dataFrame = _append_node2vec_features(G, dataFrame=dataFrame, p=1, q=1)

    return dataFrame

def _append_node2vec_features(G, dataFrame, p,q, dimensions=64):
    """
    Génère les embeddings Node2Vec et retourne un dictionnaire {node_id: vector}
    """
    print(f"Génération des marches aléatoires (dim={dimensions})...")
    
    # Configuration de Node2Vec
    # p=1, q=1 -> équivalent à DeepWalk
    # p=1, q=2 -> Favorise l'exploration locale (structure)
    # p=2, q=0.5 -> Favorise l'exploration lointaine (communautés) - homophilie
    node2vec = Node2Vec(G, 
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
    embeddings = {str(node): model.wv[str(node)] for node in G.nodes()}

    def _get_cosine_sim(u, v):
        key_u = str(int(float(u))) 
        key_v = str(int(float(v)))
        vec_u = embeddings[key_u].reshape(1, -1)
        vec_v = embeddings[key_v].reshape(1, -1)
        return cosine_similarity(vec_u, vec_v)[0][0]

    def _get_l2_dist(u, v):
        key_u = str(int(float(u))) 
        key_v = str(int(float(v)))
        vec_u = embeddings[key_u]
        vec_v = embeddings[key_v]
        return np.linalg.norm(vec_u - vec_v)

    print("Calcul des distances vectorielles pour chaque paire...")

    dataFrame[f'n2v_p{p}_q{q}_cosine'] = dataFrame.apply(lambda row: _get_cosine_sim(row['u'], row['v']), axis=1)
    dataFrame[f'n2v_p{p}_q{q}_dist'] = dataFrame.apply(lambda row: _get_l2_dist(row['u'], row['v']), axis=1)
    
    return dataFrame


########################################
# FONCTIONS D'APPEL DE SHAP ############
########################################
def analyze_with_shap(model, X_test, output_dir="outputs/plots"):
    """Calcule les SHAP values et génère les plots globaux proprement."""
    # 1. Configuration de l'explainer 'Boîte Noire' (le plus stable sur mon Mac)
    # On définit la fonction de prédiction (proba de la classe 1)
    model_predict = lambda x: model.predict_proba(x)[:, 1]
    
    # Utilisation d'un masker (échantillon de référence)
    # On prend 50 lignes pour équilibrer vitesse et précision
    masker = X_test.iloc[:50]
    
    # Initialisation de l'explainer
    explainer = shap.Explainer(model_predict, masker)    
    
    # 2. Calcul effectif des SHAP values
    # On récupère l'objet 'Explanation' complet
    shap_explanation = explainer(X_test)
    
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

def analyse_with_shap_custom(model, X_test, X_train, baseline="general", output_dir="outputs/plots"):
    groupes = {
        "Groupe_Structure": ['cn', 'aa', 'jc', 'pa', 'sp', 'pr_u', 'pr_v', 'lcc_u', 'lcc_v', 'and_u', 'and_v', 'dc_u', 'dc_v'],
        #"Groupe_Communities": ['sbm_community_u', 'sbm_community_v', 'same_csbm_community_u', 'infomap_u', 'infomap_v', 'same_infomap',"louvain_u","louvain_v","same_louvain"],
        "Groupe_Communities": ['group_u', 'group_v',  'same_group'],
        "Groupe_Embeddings": ['n2v_p2_q0.5_cosine', 'n2v_p2_q0.5_dist', 'n2v_p1_q1_cosine', 'n2v_p1_q1_dist']
    }
    
    if baseline=="CaseByCase" : 
        print("Custom SHAP baseline : case by case")
        baseline_map = {
            # Zéro pour la structure : 
            "cn": 0, "aa": 0, "pa": 0,  
            # Moyenne pour le continu :
            "n2v_p2_q0.5_cosine": X_train["n2v_p2_q0.5_cosine"].mean(),
            "n2v_p1_q1_cosine": X_train["n2v_p1_q1_cosine"].mean(),
            "n2v_p2_q0.5_dist": X_train["n2v_p2_q0.5_dist"].mean(),
            "n2v_p1_q1_dist": X_train["n2v_p1_q1_dist"].mean(),
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
    shap_values_coalition = np.zeros((len(X_test), n_groups))

    # 3. Boucle sur chaque échantillon (Sample)
    # --- Calcul des SHAP values ---
    for idx in range(len(X_test)):
        x_sample = X_test.iloc[idx]
        
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

def calculate_feature_rankings(shap_values, feature_names, output_dir="outputs/plots"):
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
    
    # Plot 3: Top 5 Appearance
    top5 = df_ranks.iloc[0:5, :].sum(axis=0).sort_values(ascending=False)
    plt.figure(figsize=(12, 7))
    sns.barplot(x=top5.index, y=top5.values, palette="viridis")
    plt.title("Importance structurelle : % de présence dans le Top 5 SHAP")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_top5_frequency.png"))
    plt.close()
    
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

def load_dataset(filename="dataset"):
    input_dir = os.path.join(PROJECT_ROOT, "outputs", "results")
    input_path = os.path.join(input_dir, filename)
    
    if not os.path.exists(input_path):
        print(f"Erreur : Le fichier n'existe pas : {input_path}")
        return None
    
    dataset = pd.read_parquet(input_path)
    print(f" Dataset chargé avec succès depuis : {input_path}")
    print(f" Taille : {dataset.shape[0]} lignes, {dataset.shape[1]} colonnes.")
    
    return dataset

def loadsave_data_joblib(data=None, filename="data.joblib", mode="save"):
    """
    Gère la sauvegarde et le chargement d'objets en .joblib (SHAP, XGBoost, etc.).
    """
    base_path = Path(PROJECT_ROOT) if 'PROJECT_ROOT' in globals() else Path.cwd()
    target_path = base_path / "outputs" / "results" / filename

    if mode == "save":
        if data is None:
            print("Erreur : Aucun objet fourni pour la sauvegarde.")
            return None
        
        # Création du dossier
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(data, target_path, compress=3)
        print(f"Objet sauvegardé dans : {target_path}")
        return target_path

    elif mode == "load":
        if not target_path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {target_path}")
        
        obj = joblib.load(target_path)
        print(f"Objet chargé avec succès depuis : {target_path}")
        
        return obj

def load_all_data_for_graph(Graph_name):

    dataset_w_com_and_dist = load_dataset(filename=f"dataset_w_com_and_dist_{Graph_name}")
    xgboost_data = loadsave_data_joblib(data=None, filename=f"xgboost_data_{Graph_name}.joblib", mode="load")
    shap_explainer = loadsave_data_joblib(data=None, filename=f"shap_explainer_{Graph_name}.joblib", mode="load")
    shap_analysis = loadsave_data_joblib(data=None, filename=f"shap_explainer_{Graph_name}.joblib", mode="load")

    return dataset_w_com_and_dist, xgboost_data, shap_explainer, shap_analysis