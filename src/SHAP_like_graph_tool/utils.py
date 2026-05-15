from .MetaLouvain import *

import random
import math
from re import X
import numpy as np
import pandas as pd
import networkx as nx
import networkx.algorithms.community as nx_comm
from networkx.algorithms.community import louvain_communities
import inspect
import igraph as ig
from pyvis.network import Network
import itertools
from math import factorial
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold, train_test_split
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.optimize import root_scalar, minimize_scalar, minimize
from scgravity import filter_data, create_q_bin, calculate_mass
import statsmodels.api as sm
from NEMtropy import UndirectedGraph
from NEMtropy import models_functions as mof

from node2vec import Node2Vec
from infomap import Infomap
from sknetwork.clustering import Louvain
import leidenalg as la
import graph_tool.all as gt
import shap
import os
import gc
import psutil 
import joblib
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
import json
from xgboost import XGBClassifier
import optuna
import time
import inspect



###############################################################
## CONTANTES, DONT MAPPING VERS ALGOS DE CALCUL DE MÉTRIQUES ##
###############################################################
CURRENT_FILE_PATH = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE_PATH)))

EMBEDDINGS = [] #['n2v_homophily', 'deepwalk', 'crosswalk']
COMMUNITY_ALGOS = [ #' 'infomap', 'sbm', 'leiden', 'surprise', 'significance', 
    #"spatial_leiden", "spatial_leiden_scgravity", "spatial_leiden_wrdb", 
'louvain', "spatial_louvain", "spatial_louvain_manualiter_0_20", "spatial_louvain_manualiter_0_50", "spatial_louvain_manualiter_0_80"
    #"spatial_louvain_manualreg", "spatial_louvain_scgravity","spatial_louvain_wrdb",
#"spatial_louvain_radiation"
]
METRICS_NODE = [] #[ "degree", "pr", "ppr", "lcc", "and", "dc", "katz"]

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
    G_train.add_nodes_from(G.nodes(data=True))
    G_train.add_edges_from(train_edges)
    G_train.graph.update(G.graph)

    G_eval = nx.Graph()
    G_eval.add_nodes_from(G.nodes(data=True))
    G_eval.add_edges_from(test_edges)
    G_eval.graph.update(G.graph)
    
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
        'ra': next(nx.resource_allocation_index(G_train, [(u, v)]))[2],
        'jc': next(nx.jaccard_coefficient(G_train, [(u, v)]))[2],
        'pa': next(nx.preferential_attachment(G_train, [(u, v)]))[2],
        'sp': nx.shortest_path_length(G_train, u, v) if nx.has_path(G_train, u, v) else 42
    }

    if has_edge:
        G_train.add_edge(u, v)

    for metric in METRICS_NODE:
        features[f'{metric}_u'] = nu.get(metric, 0)
        features[f'{metric}_v'] = nv.get(metric, 0)


    for algo in COMMUNITY_ALGOS:
        id_u = nu.get(f'{algo}_id')
        id_v = nv.get(f'{algo}_id')
        if id_u is None or id_v is None:
            print(f"ALERTE : Noeud u={u} ou v={v} a un ID None pour {algo} !")
            print(f"DEBUG: Attr cherché: {algo}_id | Présents dans nu: {list(nu.keys())}")
        pair = tuple(sorted((id_u, id_v)))

        features[f'{algo}_density'] = densities[algo].get(pair, 0)
        #features[f'same_{algo}'] = 1 if id_u == id_v else 0

    for emb in EMBEDDINGS:
        if emb in nu and emb in nv:
            vec_u = nu[emb].reshape(1, -1)
            vec_v = nv[emb].reshape(1, -1)
            hadamard_prod = vec_u * vec_v
            features[f'{emb}_cos'] = cosine_similarity(vec_u, vec_v)[0][0]
            features[f'{emb}_dist'] = np.linalg.norm(vec_u - vec_v)
            features[f'{emb}_dist_sq'] = np.linalg.norm(vec_u - vec_v)**2
            features[f'{emb}_had_mean'] = np.mean(hadamard_prod)
            features[f'{emb}_had_std'] = np.std(hadamard_prod)
        
    return features

def _worker_extract(u, v, target, G_train, densities):
    """
    Fonction isolée pour un processus : extrait les features d'une paire unique.
    """
    features = _extract_pair_features(G_train, u, v, densities)

    return {'u': u, 'v': v, 'target': target, **features}

def prepare_balanced_data(G, G_train, negative_ratio=10.0, GroundTruth = None, n_jobs=-2):
    """
    Prépare le dataset final en utilisant G_train pour les features
    et G pour vérifier l'existence réelle des liens (target).
    """
    total_cores = os.cpu_count() or 1
    if n_jobs < 0:
        n_jobs = max(1, total_cores + n_jobs)
    else:
        n_jobs = min(n_jobs, total_cores) if n_jobs > 0 else total_cores

    all_edges = list(G.edges())
    nodes = list(G.nodes())
    n_pos = len(all_edges)
    densities = prepare_all_densities(G_train)

    print(f"Préparation des listes de paires...")
    tasks = [(u, v, 1) for u, v in all_edges]
    
    n_neg_target = int(n_pos * negative_ratio)
    neg_count = 0
    while neg_count < n_neg_target:
        u, v = random.sample(nodes, 2)
        if u != v and not G.has_edge(u, v) and not G_train.has_edge(u, v):
            tasks.append((u, v, 0))
            neg_count += 1

    print(f"Extraction parallèle sur {len(tasks)} paires (n_jobs={n_jobs})...")
    
    results = Parallel(n_jobs=n_jobs, batch_size=1000, backend="loky")(
        delayed(_worker_extract)(u, v, target, G_train, densities) 
        for u, v, target in tasks
    )

    df = pd.DataFrame(results)

    for emb in EMBEDDINGS:
        dist_col = f'{emb}_dist'
        df[f'{emb}_rank'] = df[dist_col].rank(pct=True)
        
    if GroundTruth is not None:
        print(f"Injection de la Ground Truth ({len(GroundTruth)} sources)...")
        node_list = list(G.nodes()) # L'ordre utilisé lors de la création de GT_pos
        mapping = {node_id: i for i, node_id in enumerate(node_list)}
        
        indices_u = df['u'].map(mapping).values.astype(int)
        indices_v = df['v'].map(mapping).values.astype(int)
        
        for feat_name, data in GroundTruth.items():
            # Cas spécifiques (nominatifs) 
            if feat_name == 'GT_pos':
                pos_u = data[indices_u]
                pos_v = data[indices_v]
                #df['GT_pos_dist'] = np.linalg.norm(pos_u - pos_v, axis=1)
                eucl = np.linalg.norm(pos_u - pos_v, axis=1)
                R = np.linalg.norm(pos_u[0])
                df['GT_pos_dist'] = 2 * R * np.arcsin(np.clip(eucl / (2 * R), 0, 1))
                
                deg_spatial = GroundTruth.get('GT_degrees_spatial')
                if deg_spatial is not None :
                    ku = deg_spatial[indices_u]
                    kv = deg_spatial[indices_v]
                    df['GT_degrees_spatial_u'] = ku
                    df['GT_degrees_spatial_v'] = kv
                    df['GT_spatial_deg_product'] = ku * kv
                    #df['GT_spatial_gravity_log'] = (np.log(ku + 1e-5) + np.log(kv + 1e-5) - np.log(df['GT_pos_dist'] + 1e-6))

                deg_sbm = GroundTruth.get("GT_degrees_sbm")
                if deg_sbm is not None :
                    ku = deg_sbm[indices_u]
                    kv = deg_sbm[indices_v]
                    df['GT_degrees_sbm_u'] = ku
                    df['GT_degrees_sbm_v'] = kv
                    df['GT_sbm_deg_product'] = ku * kv
                    
            
            elif feat_name == 'GT_sbm_matrix':
                ids_u = GroundTruth['GT_sbm_id'][indices_u]
                ids_v = GroundTruth['GT_sbm_id'][indices_v]
                df['GT_sbm_density'] = data[ids_u, ids_v]
                
            # Cas 1 : Matrice de Paires (N x N)
            elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[0]==data.shape[1] and data.shape[0] > 100: 
                df[feat_name] = data[indices_u, indices_v]

            # Cas 2 : Vecteurs de Nœuds (N,) -> Ex: GT_degrees_sbm, GT_degrees_spatial
            elif isinstance(data, np.ndarray) and data.ndim == 1:
                df[f"{feat_name}_u"] = data[indices_u]
                df[f"{feat_name}_v"] = data[indices_v]

        print(f"DataFrame enrichi. Colonnes GT : {[c for c in df.columns if c.startswith('GT_')]}")

    print(f"DataFrame créé avec succès : {df.shape[0]} lignes.")
    return df


### Fonction de calcul de features de structure des noeuds
def computeStructureFeatures(G_train):
    print("\n--- Enrichissement du Graphe avec les Métriques de Structure ---")
    adj = nx.to_scipy_sparse_array(G_train, dtype=float)
    lambda_max = eigsh(adj, k=1, which='LM', return_eigenvectors=False)[0]
    alpha_safe = 0.85 / lambda_max

    print("Calcul : PageRank, Clustering, Average Neighbor Degree, Degree Centrality, Katz")
    degree = dict(G_train.degree())
    pr = nx.pagerank(G_train)
    ppr = nx.pagerank(G_train, alpha=0.5)
    lcc = nx.clustering(G_train)
    avg_nd = nx.average_neighbor_degree(G_train)
    dc = nx.degree_centrality(G_train)
    katz = nx.katz_centrality(G_train, alpha=alpha_safe, max_iter=1000)


    for node in G_train.nodes():
        G_train.nodes[node].update({
            'degree' : degree.get(node, 0),
            'pr': pr.get(node, 0),
            "ppr": ppr.get(node, 0),
            'lcc': lcc.get(node, 0),
            'and': avg_nd.get(node, 0),
            'dc': dc.get(node, 0),
            'katz': katz.get(node,0)
        })
    
    return G_train

#############################################
## FONCTIONS POUR INFERENCE DE COMMUNAUTES ##
#############################################

def is_partition_robust(G, partition_dict, K_min=3, min_edge_ratio=0.01):
    """
    Vérifie si la partition contient au moins K_min communautés 'significatives' en termes de nombre de liens internes (%age du nb de liens totaux du graphe)
    """
    community_edge_counts = {}
    total_edges = G.number_of_edges()
    min_edges = total_edges * min_edge_ratio
    
    for comm_id in set(partition_dict.values()):
        community_edge_counts[comm_id] = 0
        
    for u, v in G.edges():
        if partition_dict[u] == partition_dict[v]:
            community_edge_counts[partition_dict[u]] += 1
            
    robust_commus = [count for count in community_edge_counts.values() if count >= min_edges]
    
    return len(robust_commus) >= K_min

def calculate_surprise(G, partition_dict):
    """
    Calcule la Surprise d'une partition. Plus le score est élevé, plus la partition est statistiquement significative.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if m == 0: return 0

    # 1. Nombre total de paires possibles dans le graphe (M)
    M = n * (n - 1) / 2
    
    # 2. Calculer p (arêtes internes) et P (paires internes possibles)
    p = 0
    P = 0
    
    com_to_nodes = {}
    for node, com in partition_dict.items():
        com_to_nodes.setdefault(com, []).append(node)
    
    for nodes in com_to_nodes.values():
        ni = len(nodes)
        if ni < 2: continue
        
        # On extrait le sous-graphe pour compter les arêtes internes
        sub = G.subgraph(nodes)
        p += sub.number_of_edges()
        
        # Paires possibles dans cette communauté : ni * (ni-1) / 2
        P += ni * (ni - 1) / 2

    if P <= 0 or P >= M:
        return 0

    # 3. Calcul de la Surprise via l'approximation KL
    # x : densité d'arêtes interne
    # y : densité de paires interne (attendu)
    x = p / m
    y = P / M

    # Formule : m * KL(x || y)
    # KL(x||y) = x*log(x/y) + (1-x)*log((1-x)/(1-y))
    try:
        surprise = m * (x * math.log(x / y) + (1 - x) * math.log((1 - x) / (1 - y)))
    except (ValueError, ZeroDivisionError):
        return 0
    
    return surprise


def _find_best_partition(G, partition_func, K_min=3, min_edge_ratio=0.01, resolutions=None, **kwargs):
    """
    Explore les résolutions et s'arrête dès que la condition K_min est remplie.
    """
    if resolutions is None:
        resolutions = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]

    null_model = kwargs.get('null_model', None)
    sig = inspect.signature(partition_func)
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    
    best_surprise_backup = -1.0
    best_surprise = -1.0
    best_overall_partition = None
    best_overall_partition_backup = None
    best_res = -1.0
    best_res_backup = -1.0 
    
    for res in resolutions:
        communities_raw = partition_func(G, resolution=res, **filtered_kwargs)
        
        # 2. UNIFICATION DU FORMAT -> On veut un dictionnaire {node: com_id}
        if isinstance(communities_raw, dict):
            partition_dict = communities_raw
        else:
            # Si c'est une liste de sets (cas de nx.louvain_communities)
            partition_dict = {}
            for i, community in enumerate(communities_raw):
                for node in community:
                    partition_dict[node] = i

        num_commus = len(set(partition_dict.values()))
        print(f"RES LOGS - ({num_commus} commus inférées pour res = {res:.2f})")

        curr_surprise = calculate_surprise(G, partition_dict)
        
        if is_partition_robust(G, partition_dict, K_min=K_min, min_edge_ratio=min_edge_ratio):            
            if curr_surprise > best_surprise:
                best_surprise = curr_surprise
                best_overall_partition = partition_dict.copy()
                best_res = res
        else : 
            if curr_surprise > best_surprise_backup:
                best_surprise_backup = curr_surprise
                best_overall_partition_backup = partition_dict.copy()
                best_res_backup = res
            

    if best_overall_partition is None : 
        print(f"Attention : Critère K_min={K_min} non satisfait. Retour de la meilleure surprise ({best_surprise_backup:.3f})")
        best_overall_partition = best_overall_partition_backup

    print(f" Meilleure résolution : {best_res}")
    
    return best_overall_partition
    

def _appendLouvainCommunities(G_train, K_min=3, min_edge_ratio=0.01):
    best_p = _find_best_partition(
        G_train, 
        nx.community.louvain_communities, 
        K_min=K_min, 
        min_edge_ratio=min_edge_ratio,
    )
    
    nx.set_node_attributes(G_train, best_p, "louvain_id")
    _normalize_community_assignment(G_train, "louvain_id")
    
    return G_train


def _appendLeidenCommunities(G_train):
    nodes_list = list(G_train.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes_list)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G_train.edges()]
    ig_g = ig.Graph(len(nodes_list), edges)

    part_leiden = la.find_partition(ig_g, la.ModularityVertexPartition, seed=42)
    leiden_id = {nodes_list[i]: cluster for i, cluster in enumerate(part_leiden.membership)}
    nx.set_node_attributes(G_train, leiden_id, "leiden_id")
    _normalize_community_assignment(G_train, "leiden_id")
    
    return G_train

def _appendSurpriseCommunities(G_train):
    nodes_list = list(G_train.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes_list)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G_train.edges()]
    ig_g = ig.Graph(len(nodes_list), edges)

    part_surprise = la.find_partition(ig_g, la.SurpriseVertexPartition, seed=42)
    surprise_id = {nodes_list[i]: cluster for i, cluster in enumerate(part_surprise.membership)}
    nx.set_node_attributes(G_train, surprise_id, "surprise_id")
    _normalize_community_assignment(G_train, "surprise_id")
    
    return G_train

def _appendSignificanceCommunities(G_train):
    nodes_list = list(G_train.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes_list)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G_train.edges()]
    ig_g = ig.Graph(len(nodes_list), edges)
    
    part_significance = la.find_partition(ig_g, la.SignificanceVertexPartition, seed=42)
    significance_id = {nodes_list[i]: cluster for i, cluster in enumerate(part_significance.membership)}
    nx.set_node_attributes(G_train, significance_id, "significance_id")
    _normalize_community_assignment(G_train, "significance_id")
    
    return G_train

def _appendInfomapCommunities(G_train):
    im = Infomap("--two-level --silent")
    
    nodes_list = list(G_train.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes_list)}
    idx_to_node = {i: node for i, node in enumerate(nodes_list)}
    
    for source, target in G_train.edges():
        im.add_link(node_to_idx[source], node_to_idx[target])
    
    im.run()

    node_to_infomap = {node: -1 for node in nodes_list} 
    for node in im.tree:
        if node.is_leaf and node.node_id in idx_to_node:
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


def _appendSpatialLeidenCommunities(G_train, pos_attr="GT_pos", attr_name = "spatial_leiden_id",NullModel_method = "Manual_iterative"):
    if NullModel_method == "Manual" : 
        P, nodes = get_gravity_null_model_manual(G_train, pos_attr)
    elif NullModel_method == "Manual_Iterative":
        P, nodes = get_gravity_null_model_manual_iterative(G_train, pos_attr)
    elif NullModel_method == "WithReelDegreesBiais":
        P, nodes = get_gravity_null_model(G_train, pos_attr)
    elif NullModel_method == "Radiation":
        P, nodes = get_radiation_null_model_iterative(G_train, pos_attr)
    else : 
        P, nodes = optimize_scgravity_model(G_train, pos_attr)
    A = nx.to_numpy_array(G_train)
    # B = matrice de modularité débiaisée
    B = A - P
    B_symetric = (B + B.T) / 2

    asymmetry_sum = np.sum(np.abs(B - B_symetric))
    max_diff = np.max(np.abs(B - B_symetric))

    print(f"--- ANALYSE DE L'ASYMÉTRIE ---")
    print(f"Somme de la valeur absolue des différences (|B_avant - B_après|) : {asymmetry_sum:.2e}")
    print(f"Écart maximal ponctuel : {max_diff:.2e}")

    g_leiden = ig.Graph.Weighted_Adjacency(B_symetric.tolist(), mode="undirected")
    
    # On passe directement la matrice de modularité à Leiden. Equivalent sur 1ère iter, 
    # discutable sur les suivantes mais approximation à priori OK
    partition = la.find_partition(g_leiden, la.CPMVertexPartition, weights='weight', resolution_parameter=0)
    
    labels = partition.membership
    node_to_community = {nodes[i]: int(labels[i]) for i in range(len(nodes))}
    nx.set_node_attributes(G_train, node_to_community, attr_name)
    
    return G_train

def _appendSpatialLouvainCommunities(G_train, pos_attr="GT_pos", attr_name = "spatial_louvain_id", NullModel_method = "ManualIter",  K_min=3, min_edge_ratio=0.01):
    if NullModel_method == "ManualReg" : 
        P, nodes = get_gravity_null_model_manual(G_train, pos_attr)
    elif NullModel_method == "ManualIter":
        P, nodes = get_gravity_null_model_manual_iterative(G_train, pos_attr)
    elif NullModel_method == "ManualIter_0_20":
        P, nodes = get_gravity_null_model_manual_iterative(G_train, pos_attr)

        degrees = np.array([d for n, d in G_train.degree(nodes)])
        m2 = np.sum(degrees)
        P_config = np.outer(degrees, degrees) / m2
        P = (0.2 * P) + (0.8 * P_config)
    elif NullModel_method == "ManualIter_0_50":
        P, nodes = get_gravity_null_model_manual_iterative(G_train, pos_attr)

        degrees = np.array([d for n, d in G_train.degree(nodes)])
        m2 = np.sum(degrees)
        P_config = np.outer(degrees, degrees) / m2
        P = (0.5 * P) + (0.5 * P_config)
    elif NullModel_method == "ManualIter_0_80":
        P, nodes = get_gravity_null_model_manual_iterative(G_train, pos_attr)

        degrees = np.array([d for n, d in G_train.degree(nodes)])
        m2 = np.sum(degrees)
        P_config = np.outer(degrees, degrees) / m2
        P = (0.8 * P) + (0.2 * P_config)
    elif NullModel_method == "WithReelDegreesBiais":
        P, nodes = get_gravity_null_model(G_train, pos_attr)
    elif NullModel_method == "scgravity": 
        P, nodes = optimize_scgravity_model(G_train, pos_attr)
    A = nx.to_numpy_array(G_train)
    P_symetric = (P + P.T) / 2

    G_train.graph[f'P_Null_model_{NullModel_method}'] = P_symetric

    asymmetry_sum = np.sum(np.abs(P - P_symetric))
    max_diff = np.max(np.abs(P - P_symetric))

    print(f"--- ANALYSE DE L'ASYMÉTRIE ---")
    print(f"Somme de la valeur absolue des différences (|B_avant - B_après|) : {asymmetry_sum:.2e}")
    print(f"Écart maximal ponctuel : {max_diff:.2e}")

    mapping = {node: i for i, node in enumerate(nodes)}

    def my_matrix_null_model(u, v):
        idx_u = mapping[u]
        idx_v = mapping[v]
        return P_symetric[idx_u, idx_v]

    # Appel de l'algorithme développé dans MetaLouvain.py, dans la loop qui cherche la best partition
    partition = _find_best_partition(
        G_train, 
        best_partition, 
        K_min=K_min, 
        min_edge_ratio=min_edge_ratio,
        null_model=my_matrix_null_model
    )
    
    print("--- Diagnostic de l'objet partition ---")
    print(f"Nombre de nœuds assignés : {len(partition)}")
    print(f"Nombre de communautés trouvées : {len(set(partition.values()))}")
    print("---------------------------------------")

    nx.set_node_attributes(G_train, partition, attr_name)
    
    return G_train

def _appendSpatialLeidenCommunities_scgravity(G_train, pos_attr="GT_pos"):
    G_train = _appendSpatialLeidenCommunities(G_train, pos_attr=pos_attr, attr_name = "spatial_leiden_scgravity_id" ,NullModel_method = "scgravity")

def _appendSpatialLeidenCommunities_WithReelDegreesBiais(G_train, pos_attr="GT_pos"):
    G_train = _appendSpatialLeidenCommunities(G_train, pos_attr=pos_attr, attr_name = "spatial_leiden_wrdb_id" , NullModel_method = "WithReelDegreesBiais")

def _appendSpatialLouvainCommunities_scgravity(G_train, pos_attr="GT_pos"):
    G_train = _appendSpatialLouvainCommunities(G_train, pos_attr=pos_attr, attr_name = "spatial_louvain_scgravity_id" , NullModel_method = "scgravity")

def _appendSpatialLouvainCommunities_WithReelDegreesBiais(G_train, pos_attr="GT_pos"):
    G_train = _appendSpatialLouvainCommunities(G_train, pos_attr=pos_attr, attr_name = "spatial_louvain_wrdb_id" , NullModel_method = "WithReelDegreesBiais")

def _appendSpatialLouvainCommunities_ManualReg(G_train, pos_attr="GT_pos"):
    G_train = _appendSpatialLouvainCommunities(G_train, pos_attr=pos_attr, attr_name = "spatial_louvain_manualreg_id" , NullModel_method = "ManualReg")

def _appendSpatialLouvainCommunities_radiation(G_train, pos_attr="GT_pos"):
    G_train = _appendSpatialLouvainCommunities(G_train, pos_attr=pos_attr, attr_name = "spatial_louvain_radiation_id" , NullModel_method = "Radiation")

def _appendSpatialLouvainCommunities_ManualIter_0_20(G_train, pos_attr="GT_pos"):
    G_train = _appendSpatialLouvainCommunities(G_train, pos_attr=pos_attr, attr_name = "spatial_louvain_manualiter_0_20_id" , NullModel_method = "ManualIter_0_20")

def _appendSpatialLouvainCommunities_ManualIter_0_50(G_train, pos_attr="GT_pos"):
    G_train = _appendSpatialLouvainCommunities(G_train, pos_attr=pos_attr, attr_name = "spatial_louvain_manualiter_0_50_id" , NullModel_method = "ManualIter_0_50")

def _appendSpatialLouvainCommunities_ManualIter_0_80(G_train, pos_attr="GT_pos"):
    G_train = _appendSpatialLouvainCommunities(G_train, pos_attr=pos_attr, attr_name = "spatial_louvain_manualiter_0_80_id" , NullModel_method = "ManualIter_0_80")


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


def get_gravity_null_model(G, pos_attr='pos'):
    """
    Infère un modèle gravitaire (PPML) via Statsmodels.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    
    adj = nx.to_numpy_array(G, nodelist=nodes)
    degrees = np.array([d for _, d in G.degree(nodes)], dtype=float)
    coords = np.array([G.nodes[n][pos_attr] for n in nodes])
    
    dist_matrix = cdist(coords, coords, metric='euclidean')
    mass_matrix = np.outer(degrees, degrees)
    
    iu = np.triu_indices(n, k=1)
    
    y = adj[iu]
    dists = dist_matrix[iu]
    masses = mass_matrix[iu]
    
    # Préparation du DataFrame pour la régression
    df = pd.DataFrame({
        'y': y,
        'log_d': np.log(np.where(dists == 0, np.min(dists[dists > 0]) * 0.1, dists)),
        'log_m': np.log(np.where(masses == 0, 1e-9, masses))
    })
    
    # Inférence du modèle GLM Poisson (PPML) pour le modèle de gravité
    X = sm.add_constant(df[['log_m', 'log_d']])
    model = sm.GLM(df['y'], X, family=sm.families.Poisson()).fit()
    
    # Extraction des paramètres
    intercept = model.params['const']
    beta_mass = model.params['log_m']
    gamma_dist = model.params['log_d']
    
    print(f"--- Modèle Gravitaire Inféré ---")
    print(f"Friction distance (gamma) : {gamma_dist:.4f}")
    print(f"Influence masses (beta)   : {beta_mass:.4f}")
    print(f"Cte de noramlisation (intercept)   : {intercept:.4f}")
    
    # Reconstruction de la matrice de probabilité Pij = exp(intercept + beta*log_m + gamma*log_d)
    m_log_full = np.log(np.where(mass_matrix == 0, 1e-9, mass_matrix))
    d_log_full = np.log(np.where(dist_matrix == 0, 1e-9, dist_matrix))
    
    P = np.exp(intercept + beta_mass * m_log_full + gamma_dist * d_log_full)
    np.fill_diagonal(P, 0)
    
    # Vérification : comparaison à lespérance de lien 
    A_sum = len(G.edges())
    P_expected_sum = P.sum() / 2
    normalization_factor = A_sum / P_expected_sum
    
    print(f"Vérification : Nb_liens du graphe VS espérance du Null Model = {normalization_factor:.6f}")
    
    return P, nodes

def optimize_scgravity_model(G, pos_attr, target=1.0, tol=0.01, max_iter=5):
    """
    Optimise min_weight pour que normalization_factor ≈ target (≈1)
    """

    min_weight = 1e-4
    last_factor = None
    iter = 0

    for i in range(max_iter):
        _, _, factor = get_gravity_null_model_scgravity(G, pos_attr, min_weight=min_weight)
        last_factor = factor
        iter = i

        if abs(factor - target) < tol:
            break

        min_weight *= factor

    print(f"min_weight final: {min_weight:.2e} |  nb iter = {iter}")

    P, nodes, final_factor = get_gravity_null_model_scgravity(G, pos_attr, min_weight=min_weight, speak=True)

    return P, nodes
    
def get_gravity_null_model_scgravity(G, pos_attr, weight_attr='weight', min_weight = 1e-4, speak = False, nb_bins_target = 15):
    """
    Calcule la matrice du modèle nul spatial (Pij) à partir d'un graphe NetworkX.
    Suppose que les nœuds ont des attributs 'pos' (tuple ou liste [x, y]).
    """
    nodes = list(G.nodes())
    pos = nx.get_node_attributes(G, pos_attr)

    # Extraction des poids des arrêtes si il y en a, 1 sinon
    od_data = {}
    for u in nodes:
        od_data[u] = {}
        for v in nodes:
            if u == v: 
                od_data[u][v] = 1.0

            if G.has_edge(u, v):
                weight = G[u][v].get(weight_attr, 1.0)
                od_data[u][v] = float(weight)
            else:
                od_data[u][v] = min_weight
    # Calcul des distances
    dist_data = {}
    for u in nodes:
        dist_data[u] = {}
        for v in nodes:
            if u == v: continue
            d_uv = np.linalg.norm(np.array(pos[u]) - np.array(pos[v]))
            dist_data[u][v] = d_uv

    # Pipeline scgravity : https://pypi.org/project/scgravity/
    od_data_clean = filter_data(od_data, dist_data)

    

    custom_each_num = max(10, len(G.nodes())*(len(G.nodes())-1) // (2*nb_bins_target))
    #custom_each_num = 10
    q_bin = create_q_bin(od_data_clean, dist_data, each_num=custom_each_num)
    m_in, m_out, Q_hist, Q_std = calculate_mass(od_data_clean, q_bin)

    # print(f"Q_hist : {Q_hist}")
    # for node_id in m_in.keys():
    #     diff = m_in[node_id] - m_out.get(node_id)
    #     print(f"Node {node_id}| Diff: {diff}")
    #print(f"custom_each_num : {custom_each_num}")
    #print(f"od_data_clean : {od_data_clean}")
    #print(f"q_bin : {q_bin}")

    thresholds = list(q_bin['bin_left']) + [q_bin['bin_right'][-1]]
    n = len(nodes)
    m_out_vec = np.array([m_out.get(u, 0) for u in nodes])
    m_in_vec = np.array([m_in.get(u, 0) for u in nodes])
    P = np.zeros((n, n))
    
    call_dic = q_bin.get("call_dic", {})

    for i, u in enumerate(nodes):
        u_str = str(u) 
        u_bins = call_dic.get(u_str, {})
        
        for j, v in enumerate(nodes):
            if i == j: continue
            v_str = str(v)
            
            bin_idx = u_bins.get(v_str)
            
            if bin_idx is None:
                # IMPUTATION : La paire n'avait pas de lien, on cherche son bin théorique
                d_uv = dist_data[u][v]
                # np.searchsorted trouve où d_uv s'insère dans les thresholds
                idx = np.searchsorted(thresholds, d_uv) - 1
                bin_idx = max(0, min(idx, len(Q_hist) - 1))
            
            q_val = Q_hist[bin_idx]
            P[i, j] = m_out_vec[i] * m_in_vec[j] * q_val

    # Normalisation pour que la somme de P soit égale à la somme de A
    A_sum = len(G.edges())
    normalization_factor = A_sum / P.sum()
    P = P * (A_sum / P.sum())
    if speak :
        print(f"Null Model inféré normalisé par un facteur de {normalization_factor}")

    return P, nodes, normalization_factor


def get_gravity_null_model_nemtropy(G, pos_attr='pos', speak=False):
    """
    Calcule la matrice de probabilités P du modèle nul gravitaire spatial ac la lib NEMtropy
    """
    adj_obs = nx.to_numpy_array(G).astype(np.float64)
    nodes_list = list(G.nodes())
    n_nodes = len(nodes_list)
    obs_edges_count = G.number_of_edges()
    obs_degrees = adj_obs.sum(axis=1).astype(np.float64)
    obs_degrees[obs_degrees == 0] = 1e-10
    
    positions = np.array([G.nodes[n][pos_attr] for n in nodes_list])
    dist_matrix = squareform(pdist(positions)).astype(np.float64)
    dist_matrix = np.exp(-1.0 * dist_matrix)
    np.fill_diagonal(dist_matrix, 0)
    
    graph = UndirectedGraph(adjacency=adj_obs)

    graph.strength_sequence = obs_degrees

    graph.solve_tool(model='crema-sparse', adjacency=dist_matrix, method='fixed-point',tol=1e-04,
                         full_return=True, max_steps=2000, verbose=False)

    print("--- Inspection de l'objet UndirectedGraph ---")
    attrs = [attr for attr in dir(graph) if not attr.startswith('__') and not callable(getattr(graph, attr))]
    for attr in attrs:
        val = getattr(graph, attr)
        # On affiche un résumé pour ne pas flooder la console
        summary = f"{type(val)} (shape/len: {getattr(val, 'shape', len(val) if hasattr(val, '__len__') else 'N/A')})"
        print(f"{attr}: {summary}")

    print("\n--- Diagnostic des arguments du solveur ---")
    if hasattr(graph, 'args'):
        print(f"Nombre d'arguments dans 'args': {len(graph.args)}")

    def inspect_nemtropy_expert(graph):
        """
        Analyse profonde de l'objet UndirectedGraph pour identifier 
        la source du problème de convergence/échelle.
        """
        print("="*60)
        print("🔍 AUTOPSIE EXPERTE DU MODÈLE NEMTROPY")
        print("="*60)

        # 1. Identification de la Solution (Les multiplicateurs)
        print("\n[1] PARAMÈTRES DE LA SOLUTION (X_i)")
        sol_candidates = ['beta', 'solution_array', 'x', 'solution']
        found_sol = False
        for cand in sol_candidates:
            if hasattr(graph, cand) and getattr(graph, cand) is not None:
                val = getattr(graph, cand)
                print(f"✅ {cand:15}: Moyenne={np.mean(val):.4f}, Min={np.min(val):.4f}, Max={np.max(val):.4f}")
                found_sol = True
        if not found_sol:
            print("❌ AUCUNE SOLUTION TROUVÉE : Le solveur n'a rien enregistré.")

        # 2. Vérification des Contraintes (Cibles)
        print("\n[2] ÉTAT DES CONTRAINTES (DEGRÉS)")
        if hasattr(graph, 'strength_sequence'):
            s = graph.strength_sequence
            print(f"🎯 Cibles (k_i)   : Somme={np.sum(s)/2:.2f}, Moyenne={np.mean(s):.2f}")
        if hasattr(graph, 'expected_stregth_seq'):
            e = graph.expected_stregth_seq
            print(f"📈 Attendues (<k>): Somme={np.sum(e)/2:.2f}, Moyenne={np.mean(e):.2f}")
        
        # 3. Diagnostic de l'Erreur
        print("\n[3] DIAGNOSTIC DE CONVERGENCE")
        for err_attr in ['error', 'error_strength', 'relative_error_strength']:
            if hasattr(graph, err_attr):
                print(f"⚠️ {err_attr:23}: {getattr(graph, err_attr)}")

        # 4. Analyse des entrées (The "Hidden" Args)
        print("\n[4] STRUCTURE INTERNE (ARGS)")
        if hasattr(graph, 'args') and graph.args is not None:
            print(f"Nombre d'arguments dans 'args' : {len(graph.args)}")
            for i, arg in enumerate(graph.args):
                t = type(arg)
                m = np.mean(arg) if hasattr(arg, 'mean') else "N/A"
                print(f"   - Arg[{i}] ({t.__name__}): Moyenne={m}")

        print("\n" + "="*60)

    inspect_nemtropy_expert(graph)

    return P, graph.sol[:-1]

def get_gravity_null_model_manual(G, pos_attr='pos', speak=False):
    """
    Calcule le modèle nul gravitaire via une régression logistique dyadique. Infère conjointement le beta spatial et les fitness (alphas).
    """
    nodes = list(G.nodes())
    n = len(nodes)
    adj = nx.to_numpy_array(G) # Dim N*N
    pos = nx.get_node_attributes(G, pos_attr)
    
    rows, cols = np.triu_indices(n, k=1) # Deux vecteurs de taille M = (N * (N-1) / 2). vecteurs plats
    
    # Calcul des distances euclidiennes pour chaque paire
    pos_array = np.array([pos[u] for u in nodes]) # Forme : Matrice (N, 2)
    dist_matrix = np.linalg.norm(pos_array[:, np.newaxis] - pos_array[np.newaxis, :], axis=2) # Forme : Matrice (N, N)
    distances_flat = dist_matrix[rows, cols] # Dim M, qui va chercher sa valeur dans dist_matrix selon la valeur de row[200] par ex
    links_flat = adj[rows, cols] # Dim M

    # 2. Construction de la matrice de design X
    # On crée une colonne pour chaque noeud (alphas) et une pour la distance (beta)
    # Pour chaque paire (i, j), les colonnes i et j valent 1, les autres 0.
    num_dyads = len(links_flat)
    X = np.zeros((num_dyads, n + 1)) # Dim M, N+1
    
    # On remplit les indices des noeuds
    X[np.arange(num_dyads), rows] = 1 # Pour chaque élément de l'axe de taille M (paires possibles), 
                                      # on met un 1 uniquement dans la colonne correspondant au noeud visé
    X[np.arange(num_dyads), cols] = 1 # On obtient donc une matrice où chaque ligne cible une paire, et n'a 1
                                      # que sur chaque noeud de la paire concernée
    # On remplit la distance dans une dernière colonne (on met -dist pour que beta soit positif si dissuasion)
    X[:, -1] = -distances_flat

    # 3. Inférence par Maximum de Vraisemblance (Logit)
    if speak:
        print(f"Lancement de l'inférence pour {n} noeuds ({num_dyads} dyades)...")
    
    model = sm.Logit(links_flat, X)
    # L-BFGS est robuste pour les modèles avec beaucoup de paramètres
    result = model.fit(method='lbfgs', maxiter=1000, disp=speak)
    
    # 4. Extraction des paramètres
    alphas = result.params[:n]
    beta = result.params[-1]
    
    # 5. Reconstruction de la matrice de probabilités P
    # theta_ij = alpha_i + alpha_j - beta * d_ij
    theta = alphas[:, np.newaxis] + alphas[np.newaxis, :] - beta * dist_matrix
    P = 1 / (1 + np.exp(-theta))
    np.fill_diagonal(P, 0)
    
    if speak:
        gravity_inference_health_check(adj, P, dist_matrix)
    
    # Normalisation pour que la somme de P soit égale à la somme de A
    A_sum = len(G.edges())
    normalization_factor = 2*A_sum / P.sum()
    print(f"Vérification : Null Model donne P.sum / 2*nb_edges = {normalization_factor}")
    print(f"Alpha moyen = {np.mean(alphas):.4f}, beta = {beta:.6f}")
    
    return P, nodes

def gravity_inference_health_check(adjacency_mtx, P_mtx, dist_mtx):
        obs_degrees = adjacency_mtx.sum(axis=1)
        exp_degrees = P_mtx.sum(axis=1)
        
        corr = np.corrcoef(obs_degrees, exp_degrees)[0, 1]
        mae = np.mean(np.abs(obs_degrees - exp_degrees))
        
        upper_idx = np.triu_indices(len(obs_degrees), k=1)
        dists = dist_mtx[upper_idx]
        p_vals = P_mtx[upper_idx]
        actual_links = adjacency_mtx[upper_idx]

        print(f"--- Rapport de Cohérence ---")
        print(f"Corrélation Obs/Exp : {corr:.6f}")
        print(f"MAE sur les degrés : {mae:.4f}")
        
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Calibration
        plt.subplot(1, 3, 1)
        plt.scatter(obs_degrees, exp_degrees, alpha=0.5)
        plt.plot([obs_degrees.min(), obs_degrees.max()], [obs_degrees.min(), obs_degrees.max()], 'r--')
        plt.title("Calibration des Degrés")
        plt.xlabel("Degré Observé")
        plt.ylabel("Degré Attendu")
        
        # Plot 2: Résidus
        plt.subplot(1, 3, 2)
        plt.hist(obs_degrees - exp_degrees, bins=20)
        plt.title("Distribution des Erreurs (Obs - Exp)")
        
        # Plot 3: Distance Decay (binned)
        # Plot 3: Distance Decay (binned) avec Ratios d'effectifs
        ax3 = plt.subplot(1, 3, 3)
        bins = np.linspace(dists.min(), dists.max(), 20)
        bin_idx = np.digitize(dists, bins)
        bin_centers = bins[:-1] + np.diff(bins) / 2
        total_pairs = len(dists)
        
        obs_trend = []
        exp_trend = []
        for i in range(1, len(bins)):
            mask = (bin_idx == i)
            if np.any(mask):
                obs_trend.append(actual_links[mask].mean())
                exp_trend.append(p_vals[mask].mean())
                # Calcul du ratio pour l'affichage
                ratio = (np.sum(mask) / total_pairs)*100
                label = f"{ratio:.2g}" if ratio >= 0.1 else f"{ratio:.0e}"
                ax3.annotate(label, xy=(bin_centers[i-1], obs_trend[-1]), 
                             xytext=(0, -8), textcoords="offset points",
                             va='top', ha='center', fontsize=8, color='black', alpha=0.8)
            else:
                obs_trend.append(0); exp_trend.append(0)

        ax3.plot(bin_centers, obs_trend, 'bo-', label='Observed (Freq)')
        ax3.plot(bin_centers, exp_trend, 'rx--', label='Model (Pij)')
        ax3.set_xlabel("Distance d_ij")
        ax3.set_ylabel("Probabilité de connexion P(Lien)")
        ax3.set_title("Validation de la Friction Spatiale\n(% du total des paires sous les points)")
        ax3.legend()
       
        plt.tight_layout()
        plt.show()

def get_gravity_null_model_manual_iterative(G, pos_attr='pos', tol=0.01, max_iter=1000, speak = False):
    nodes = list(G.nodes())
    n = len(nodes)
    adj = nx.to_numpy_array(G)
    degrees = np.sum(adj, axis=1)
    
    # Matrice de distance (N, N)
    pos_array = np.array([G.nodes[u][pos_attr] for u in nodes])
    #dist_matrix = np.linalg.norm(pos_array[:, np.newaxis] - pos_array[np.newaxis, :], axis=2)
    eucl = np.linalg.norm(pos_array[:, np.newaxis] - pos_array[np.newaxis, :], axis=2)
    R = np.linalg.norm(pos_array[0]) 
    dist_matrix = 2 * R * np.arcsin(np.clip(eucl / (2 * R), 0, 1))
    print(f"DIST calculée bieng pour Airports, R = {R}")
   
    
    # Initialisation des paramètres
    alphas = np.zeros(n)
    beta = 1.0
    
    def func_alpha_i(alpha_i, other_alphas, dist_row_i, current_beta, target_k):
        """
        Calcule la valeur de f(alpha_i) et sa dérivée pour la méthode de Newton.
        f(alpha_i) = somme(probs) - k_i
        """
        theta = alpha_i + other_alphas - current_beta * dist_row_i
        probs = 1 / (1 + np.exp(-theta))
        
        val = np.sum(probs) - target_k
        # Dérivée d'une somme de sigmoïdes
        grad = np.sum(probs * (1 - probs)) # Vérifié sur papier, c'est vrai.
        return val, grad
      
    def total_log_likelihood_beta(b, current_alphas):
        """ Log-vraisemblance négative pour l'optimisation de beta """
        # theta_ij = alpha_i + alpha_j - beta * d_ij
        theta = current_alphas[:, np.newaxis] + current_alphas[np.newaxis, :] - b * dist_matrix
        # log(1 + exp(theta)) via logaddexp pour la stabilité
        log_q = np.logaddexp(0, theta)
        # On ne prend que le triangle supérieur (réseau non dirigé)
        ll = np.sum(np.triu(adj * theta - log_q, k=1))
        return -ll

    # --- BOUCLE PRINCIPALE ---

    for iteration in range(max_iter):
        old_alphas = alphas.copy()
        
        # 1. Mise à jour séquentielle des alphas (Descente de coordonnées)
        for _ in range(5):
            theta = alphas[:, np.newaxis] + alphas[np.newaxis, :] - beta * dist_matrix
            # Clipping de theta pour éviter exp(large)
            theta = np.clip(theta, -50, 50) 
            
            P = 1 / (1 + np.exp(-theta))
            np.fill_diagonal(P, 0)
            
            f_x = np.sum(P, axis=1) - degrees
            # f_prime est la Hessienne
            f_prime = np.sum(P * (1 - P), axis=1) + 1e-5
            
            # MISE À JOUR BRIDÉE : On ne bouge pas de plus de 2.0 par étape
            step = f_x / f_prime
            alphas -= 0.5 * np.clip(step, -2.0, 2.0)
            
        # 2. Mise à jour de beta
        res_beta = minimize_scalar(
            total_log_likelihood_beta, 
            args=(alphas,), 
            bounds=(0, 20), 
            method='bounded'
        )
        beta = res_beta.x
        
        # Convergence
        theta = alphas[:, np.newaxis] + alphas[np.newaxis, :] - beta * dist_matrix
        current_P = 1 / (1 + np.exp(-theta))
        np.fill_diagonal(current_P, 0)
        
        predicted_degrees = np.sum(current_P, axis=1)
        mae_degrees = np.mean(np.abs(predicted_degrees - degrees))
        
        if iteration %100 == 0:
            print(f"Iteration {iteration}: MAE = {mae_degrees:.6f}, Beta = {beta:.4f}")
        
        if mae_degrees < tol:
            break

    print(f"Modèle gravitaire inféré, avec une MAE de {mae_degrees}, alpha moy = {np.mean(alphas):.4f} et beta = {beta}")  
    A_sum = len(G.edges())
    normalization_factor = 2*A_sum / P.sum()
    print(f"Vérification : Null Model donne P.sum / 2*nb_edges = {normalization_factor}")
    if speak : 
        gravity_inference_health_check(adj, current_P, dist_matrix)   
    
    return current_P, nodes


def get_radiation_null_model_iterative(G, pos_attr='pos', tol=0.01, max_iter=1000, speak=False):
    nodes = list(G.nodes())
    n = len(nodes)
    adj = nx.to_numpy_array(G)
    degrees = np.sum(adj, axis=1)
    target_sum = np.sum(degrees)

    # 1. Pré-calcul des distances et des masques s_ij
    pos_array = np.array([G.nodes[u][pos_attr] for u in nodes])
    dist_matrix = np.linalg.norm(pos_array[:, np.newaxis] - pos_array[np.newaxis, :], axis=2)
    # mask[i, j, k] est vrai si k est plus proche de i que ne l'est j
    masks = [dist_matrix[i, :][:, np.newaxis] > dist_matrix[i, :][np.newaxis, :] for i in range(n)]

    # Variables globales pour le monitoring via callback
    iteration_data = {'count': 0, 'last_C': 0.0}

    def compute_P_and_C(z_vec):
        """Calcule la matrice P et le facteur de normalisation C."""
        z = np.exp(z_vec)
        s = np.zeros((n, n))
        for i in range(n):
            s[i, :] = np.dot(masks[i], z)
        
        zi = z[:, np.newaxis]
        zj = z[np.newaxis, :]
        denom = (zi + s) * (zi + zj + s)
        
        P_raw = np.divide(zi * zj, denom, out=np.zeros_like(s), where=denom!=0)
        np.fill_diagonal(P_raw, 0)
        
        # Calcul de C pour que Sum(P) == Sum(Degrees)
        current_sum = np.sum(P_raw)
        C = target_sum / current_sum if current_sum > 0 else 1.0
        return P_raw * C, C

    def objective(z_vec):
        """Fonction cible : on minimise la MSE sur les degrés."""
        P, C = compute_P_and_C(z_vec)
        iteration_data['last_C'] = C # Stockage pour le callback
        pred_degrees = np.sum(P, axis=1)
        # MSE est plus stable pour le gradient que la MAE
        return np.mean((pred_degrees - degrees)**2)

    def callback(z_vec):
        """Affiche les stats toutes les 10 itérations."""
        iteration_data['count'] += 1
        if iteration_data['count'] % 10 == 0:
            P, C = compute_P_and_C(z_vec)
            pred_degrees = np.sum(P, axis=1)
            mae = np.mean(np.abs(pred_degrees - degrees))
            print(f"Iteration {iteration_data['count']}: MAE = {mae:.6f}, C = {C:.4e}")

    # Initialisation : log des degrés (pour garantir z > 0)
    # On clip pour éviter log(0)
    x0 = np.log(np.clip(degrees, 1e-1, None))

    if speak:
        print(f"Lancement de l'optimisation L-BFGS-B pour {n} noeuds...")

    res = minimize(
        objective, 
        x0=x0, 
        method='L-BFGS-B',
        callback=callback,
        options={'maxiter': 200, 'ftol': 1e-7}
    )

    # Reconstruction finale
    final_P, final_C = compute_P_and_C(res.x)
    final_z = np.exp(res.x)
    final_mae = np.mean(np.abs(np.sum(final_P, axis=1) - degrees))

    print(f"\n--- Modèle de Radiation Optimisé ---")
    print(f"MAE finale : {final_mae:.6f}")
    print(f"Facteur de normalisation global C : {final_C:.4e}")

    check_val = np.sum(final_P) / target_sum
    print(f"Vérification : Null Model radiation donne P.sum / 2*nb_edges = {check_val:.4f}")
    if speak :
        gravity_inference_health_check(adj, final_P, dist_matrix)
   
    return final_P, nodes

COMMUNITY_MAPPING = {
    'louvain': _appendLouvainCommunities,
    'infomap': _appendInfomapCommunities,
    'sbm': _appendGraphToolSBM,
    'leiden': _appendLeidenCommunities,
    'surprise': _appendSurpriseCommunities,
    'significance': _appendSignificanceCommunities,
    "spatial_leiden" : _appendSpatialLeidenCommunities,
    "spatial_louvain" : _appendSpatialLouvainCommunities,
    "spatial_leiden_scgravity" : _appendSpatialLeidenCommunities_scgravity,
    "spatial_louvain_scgravity" : _appendSpatialLouvainCommunities_scgravity,
    "spatial_leiden_wrdb" : _appendSpatialLeidenCommunities_WithReelDegreesBiais,
    "spatial_louvain_wrdb" : _appendSpatialLouvainCommunities_WithReelDegreesBiais,
    #"spatial_louvain_radiation" : _appendSpatialLouvainCommunities_radiation,
    "spatial_louvain_manualreg" : _appendSpatialLouvainCommunities_ManualReg,
    "spatial_louvain_manualiter_0_20" : _appendSpatialLouvainCommunities_ManualIter_0_20,
    "spatial_louvain_manualiter_0_50" : _appendSpatialLouvainCommunities_ManualIter_0_50,
    "spatial_louvain_manualiter_0_80" : _appendSpatialLouvainCommunities_ManualIter_0_80
}


def computeCommunityFeatures(G_train, algos="All", spatial_ref = "GT_pos"):
    print("\n--- Enrichissement du Graphe avec les Communautés ---")
    to_run = COMMUNITY_ALGOS if algos == "All" else algos
    
    for algo in to_run:
        if algo in COMMUNITY_MAPPING:
            print(f"Calcul des communautés via {algo}...")
            if algo.startswith("spatial_"):
                COMMUNITY_MAPPING[algo](G_train, pos_attr= spatial_ref)
            else :
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

###########################################
## FONCTIONS POUR INFERENCE D'EMBEDDINGS ##
###########################################

def _append_node2vec_features(G_train, p, q, attr_name,dimensions=64):
    """
    Génère les embeddings Node2Vec et retourne un dictionnaire {node_id: vector}
    """
    print(f"Calcul de Node2Vec (p={p}, q={q})...")
    print(f"Génération des marches aléatoires (dim={dimensions})...")

    cores = multiprocessing.cpu_count() -1
    
    # Configuration de Node2Vec
    # p=1, q=1 -> équivalent à DeepWalk
    # p=1, q=2 -> Favorise l'exploration locale (structure)
    # p=2, q=0.5 -> Favorise l'exploration lointaine (communautés) - homophilie
    node2vec = Node2Vec(G_train, 
                        dimensions=dimensions, 
                        walk_length=30, 
                        num_walks=100, 
                        workers=cores, 
                        p=p, q=q)

    print("Entraînement du modèle Skip-gram...")
    start_skip = time.time()
    try:
        model = node2vec.fit(window=10, min_count=1, batch_words=1000, vector_size=dimensions, workers=cores)
    except TypeError:
        model = node2vec.fit(window=10, min_count=1, batch_words=1000, size=dimensions, workers=cores)
    
    # On récupère les vecteurs dans un dictionnaire
    embeddings = {node: model.wv[str(node)] for node in G_train.nodes()}
    nx.set_node_attributes(G_train, embeddings, attr_name)

    end_skip = time.time()
    skipgram_duration = end_skip - start_skip
    print(f"Skip-gram terminé en {skipgram_duration:.2f}s")

def _apply_crosswalk_weights(G_train, group_attr='sbm_id', alpha=0.5, p_bound=1.0, walk_length_d=5, num_walks_r=10):
    print(f"Application du biais Crosswalk (alpha={alpha})...")
    print(f"Calcul de m(v) via {num_walks_r} marches de longueur {walk_length_d}...")
    
    m = {}
    nodes = list(G_train)
    
    # 1 - Calcul de m(v) basé sur r rdm wlaks de taille d
    for v in nodes:
        lv = G_train.nodes[v][group_attr]
        total_cross_group_visits = 0
        
        for _ in range(num_walks_r):
            current_node = v
            for _ in range(walk_length_d):
                neighbors = list(G_train.neighbors(current_node))
                if not neighbors:
                    break
                
                weights = [G_train[current_node][nbr].get('weight', 1.0) for nbr in neighbors]
                current_node = random.choices(neighbors, weights=weights, k=1)[0]
                
                if G_train.nodes[current_node][group_attr] != lv:
                    total_cross_group_visits += 1
        
        m[v] = total_cross_group_visits / (num_walks_r * walk_length_d)
        
        if m[v] == 0:
            m[v] = 1e-6

    # 2 - Repondération des arêtes
    new_weights = {}
    for v in G_train.nodes():
        neighbors = list(G_train.neighbors(v))
        if not neighbors: continue
        
        lv = G_train.nodes[v][group_attr]
        
        same_group_neighbors = [u for u in neighbors if G_train.nodes[u][group_attr] == lv]
        diff_group_neighbors = [u for u in neighbors if G_train.nodes[u][group_attr] != lv]
        
        sum_same = sum(G_train.get_edge_data(v, u).get('weight', 1.0) * (m[u]**p_bound) for u in same_group_neighbors)
        sum_diff = sum(G_train.get_edge_data(v, u).get('weight', 1.0) * (m[u]**p_bound) for u in diff_group_neighbors)
        
        # Attribution des nouveaux poids
        for u in neighbors:
            w_vu = G_train.get_edge_data(v, u).get('weight', 1.0)
            lu = G_train.nodes[u][group_attr]
            
            if lu == lv:
                new_w = (1 - alpha) * w_vu * (m[u]**p_bound) / sum_same if sum_same > 0 else w_vu
            else:
                num_diff_groups = len(set(G_train.nodes[z][group_attr] for z in diff_group_neighbors))
                new_w = (alpha * w_vu * (m[u]**p_bound)) / (num_diff_groups * sum_diff) if sum_diff > 0 else w_vu
            
            new_weights[(v, u)] = new_w

    # 3 - MAJ du graphe
    nx.set_edge_attributes(G_train, new_weights, 'weight')
    print("Repondération terminée.")

    return G_train

def _append_crosswalk_features(G_train, p, q, attr_name, dimensions=64):
    G_weighted = G_train.copy()

    _append_node2vec_features(G_weighted, p=p, q=q,  attr_name=attr_name, dimensions=dimensions)
    embeddings = nx.get_node_attributes(G_weighted, attr_name)
    nx.set_node_attributes(G_train, embeddings, attr_name)

    
EMBEDDING_MAPPING = {
    'n2v_homophily': lambda G: _append_node2vec_features(G, p=2, q=0.5, attr_name="n2v_homophily"),
    'deepwalk': lambda G: _append_node2vec_features(G, p=1, q=1, attr_name="deepwalk"),
    'crosswalk': lambda G: _append_crosswalk_features(G, p=1, q=1, attr_name="crosswalk")
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


#################################################
######### FONCTIONS DE CROSS VALIDATION #########
#################################################

def k_fold_cross_validation(G, k=2, features_list=None, n_trials=50, GroundTruth =None, graph_name="G_NAME"):
    
    folds_data = _prepare_precalculated_folds(G, k=k, GroundTruth=GroundTruth)
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

def _process_single_fold(f_idx, t_idx, v_idx, edges, nodes_data, GroundTruth=None):
    print(f"--- Démarrage Parallèle Fold {f_idx + 1} ---")
    # Construction du graphe kept
    kept_edges = [edges[i] for i in t_idx]
    G_kept = nx.Graph()
    G_kept.add_nodes_from(nodes_data)
    G_kept.add_edges_from(kept_edges)

    # Séparation en graphe de train/test
    G_train, G_test = hide_graph_links(G_kept, test_size=0.15)
    
    # G_hiden : pour le validation set
    hidden_edges = [edges[i] for i in v_idx]
    G_hidden = nx.Graph()
    G_hidden.add_nodes_from(nodes_data)
    G_hidden.add_edges_from(hidden_edges)

    # Enrichissement du graphe de train
    G_train = computeStructureFeatures(G_train)
    G_train = computeDistanceFeatures(G_train)
    G_train = computeCommunityFeatures(G_train)

    # Enrichissement du graphe de validation finale
    G_kept = computeStructureFeatures(G_kept)
    G_kept = computeDistanceFeatures(G_kept)
    G_kept = computeCommunityFeatures(G_kept)

    
    # Création des datasets
    ds_train = prepare_balanced_data(G_test, G_train, negative_ratio=10.0, GroundTruth=GroundTruth) 
    ds_val = prepare_balanced_data(G_hidden, G_kept, negative_ratio=25.0, GroundTruth=GroundTruth)
    
    return (ds_train, ds_val)

def _prepare_precalculated_folds(G, k=1, GroundTruth = None):
    edges = list(G.edges())
    nodes_data = list(G.nodes(data=True))

    if k == 1:
        folds_idx = [train_test_split(range(len(edges)), test_size=0.2, random_state=42)]
    else:
        kf = KFold(n_splits=k, shuffle=True)
        folds_idx = list(kf.split(edges))

    print(f"[K-FOLD] Préparation séquentielle de {len(folds_idx)} folds...")

    # Anciennement //isé, plus efficace comme ça pour éviter //isations imbriquées.
    precalculated_folds = [
        _process_single_fold(i, t_idx, v_idx, edges, nodes_data, GroundTruth=GroundTruth)
        for i, (t_idx, v_idx) in enumerate(folds_idx)
    ]
    
    return precalculated_folds

def _run_optuna_tuning(precalculated_folds, features_list=None, n_trials=50, n_jobs = -2):

    if features_list is None or len(features_list) == 0:
        exclude = ['u', 'v', 'target', 'label']
        features = [
            col for col in precalculated_folds[0][0].columns
            if (col not in exclude and not col.startswith('GT_'))
            #or col in ['GT_sbm_density', 'GT_pos_dist','GT_spatial_deg_product', 'GT_sbm_deg_product']
        ]
        print(f"Features détectées ({len(features)}) : {features}")
    else:
        features = features_list

    optimized_folds = []
    for ds_train, ds_val in precalculated_folds:
        optimized_folds.append({
            'X_train': ds_train[features].values.astype('float32'),
            'y_train': ds_train['target'].values,
            'X_val': ds_val[features].values.astype('float32'),
            'y_val': ds_val['target'].values
        })

    total_cores = os.cpu_count() or 1
    if n_jobs < 0:
        n_jobs = max(1, total_cores + n_jobs)
    else:
        n_jobs = min(n_jobs, total_cores) if n_jobs > 0 else total_cores

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
            "n_jobs" : n_jobs,
            'random_state': 42
        }

        f_auc_v, f_auc_t, f_ap_v = [], [], []

        """
        for ds_train, ds_val in precalculated_folds:
            model = XGBClassifier(**params)
            model.fit(ds_train[features], ds_train['target'])
            
            p_val = model.predict_proba(ds_val[features])[:, 1]
            p_train = model.predict_proba(ds_train[features])[:, 1]
            
            f_auc_v.append(roc_auc_score(ds_val['target'], p_val))
            f_auc_t.append(roc_auc_score(ds_train['target'], p_train))
            f_ap_v.append(average_precision_score(ds_val['target'], p_val))
        """

        for fold in optimized_folds:
            model = XGBClassifier(**params)
            model.fit(fold['X_train'], fold['y_train'])

            p_val = model.predict_proba(fold['X_val'])[:, 1]
            p_train = model.predict_proba(fold['X_train'])[:, 1]
            
            f_auc_v.append(roc_auc_score(fold['y_val'], p_val))
            f_auc_t.append(roc_auc_score(fold['y_train'], p_train))
            f_ap_v.append(average_precision_score(fold['y_val'], p_val))
        
        avg_auc_v = np.mean(f_auc_v)
        trial.set_user_attr("std_auc", np.std(f_auc_v))
        trial.set_user_attr("avg_ap", np.mean(f_ap_v))
        trial.set_user_attr("delta_auc", np.mean(f_auc_t) - avg_auc_v)

        del model 
        gc.collect()

        return avg_auc_v

    optuna.logging.set_verbosity(optuna.logging.WARNING)  # Pour ne garder que les logs d'erreur d'optuna
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

def analyze_with_shap_tree(model, X_test, y_test, max_pos=2000, negative_ratio=1.0):
    """
    Version optimisée via TreeExplainer pour modèles basés sur les arbres.
    """
    pos_indices = y_test[y_test == 1].index
    n_pos = min(len(pos_indices), max_pos)
    pos_sample = pos_indices[:n_pos]
    
    n_neg = int(n_pos * negative_ratio)
    neg_indices = y_test[y_test == 0].index
    neg_sample = y_test.loc[neg_indices].sample(n=n_neg, random_state=42).index
    
    X_shap = X_test.loc[pos_sample.union(neg_sample)]
    
    print(f"Calcul TreeExplainer : {len(X_shap)} échantillons au total.")

    booster = model.get_booster()
    explainer = shap.TreeExplainer(booster)

    shap_explanation = explainer(X_shap)
    
    if isinstance(shap_explanation, tuple):
        # On ne garde que l'explication pour la classe 1 (positif)
        shap_explanation = shap_explanation[1]
    else:
        shap_explanation = shap_explanation

    return shap_explanation

def display_shap(graphname, output_dir="outputs/plots"):

    filename = f"shap_explainer_{graphname}.joblib"
    shap_explainer = loadsave_data_joblib(data=None, filename=filename, mode="load")

    # --- GÉNÉRATION DES PLOTS ---
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Summary Points (Beeswarm)
    plt.figure(figsize=(12, 8))
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
        #"Groupe_Communities": ['sbm_u', 'sbm_v', 'same_sbm', 'infomap_u', 'infomap_v', 'same_infomap',"louvain_u","louvain_v","same_louvain"],
        "Groupe_Communities": ['sbm_density', 'same_sbm', 'infomap_density', 'same_infomap',"louvain_density", "same_louvain"],
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

def build_explainability_dataset(shap_explanation, xgboost_data, dataset_hidden, feature_mapping, p_matrices_origin= None, ratio_sbm = None, max_pos=2000, negative_ratio=1.0):
    """
    Construit le dataset d'analyse en réintégrant u et v depuis dataset_hidden.
    """
    y_hidden = xgboost_data['y_hidden']
    
    # Recréation de l'échantillon (identique à l'analyse SHAP)
    pos_indices = y_hidden[y_hidden == 1].index
    n_pos = min(len(pos_indices), max_pos)
    pos_sample = pos_indices[:n_pos]
    
    n_neg = int(n_pos * negative_ratio)
    neg_indices = y_hidden[y_hidden == 0].index
    neg_sample = neg_indices[:n_neg] # On garde les premiers pour la reproductibilité
    
    indices = pos_sample.union(neg_sample)
    
    analysis_df = pd.DataFrame(index=indices)
    analysis_df['u'] = dataset_hidden.loc[indices, 'u']
    analysis_df['v'] = dataset_hidden.loc[indices, 'v']
    analysis_df['target'] = dataset_hidden.loc[indices, 'target']
    
    X_hidden = xgboost_data['X_hidden'].loc[indices]
    analysis_df['proba'] = xgboost_data['model'].predict_proba(X_hidden.values)[:, 1]

    feature_names = list(shap_explanation.feature_names)
    for family, features in feature_mapping.items():
        col_indices = [feature_names.index(f) for f in features if f in feature_names]
        if col_indices:
            analysis_df[f'SHAP_{family}'] = np.abs(shap_explanation.values[:, col_indices]).sum(axis=1)
        else:
            analysis_df[f'SHAP_{family}'] = 0.0

    # 5. Calcul de l'Index de Dominance
    num = analysis_df['SHAP_Groupe_Communities'] - analysis_df['SHAP_Groupe_Embeddings']
    den = analysis_df['SHAP_Groupe_Communities'] + analysis_df['SHAP_Groupe_Embeddings'] + 1e-10
    analysis_df['Dominance_Index'] = num / den

    if p_matrices_origin is not None : 
        u_idx = analysis_df["u"].astype(int).to_numpy()
        v_idx = analysis_df["v"].astype(int).to_numpy()
        analysis_df["p_uv_sbm"] = p_matrices_origin[1.00][u_idx, v_idx]
        analysis_df["p_uv_pos"] = p_matrices_origin[0.00][u_idx, v_idx] 

        if ratio_sbm is not None :
            analysis_df["p_uv_hyb"] = p_matrices_origin[ratio_sbm][u_idx, v_idx] 
    
    return analysis_df[analysis_df['target'] == 1]
    

def plot_pyvis_eval_graph_map(explainability_dataset, G, G_name, filename="feature_mapping.html"):
    """
    Génère une visualisation physique interactive (HTML) dans le notebook.
    """
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=True, cdn_resources='remote')
    net.heading = f'<h1 style="color: #ffcc00; font-family: sans-serif; margin-left: 20px;">Graphe : {G_name}</h1>'

    for node in G.nodes():
        net.add_node(node, label=str(node), size=10, color="#555555")

    df_pos = explainability_dataset[explainability_dataset['target'] == 1].copy()

    cmap = plt.cm.coolwarm

    for idx, row in df_pos.iterrows():
        u, v = row['u'], row['v'] 
        score_norm = (row['Dominance_Index'] + 1) / 2
        color_hex = mcolors.to_hex(cmap(score_norm))
        edge_width = 2 + (row['proba'] * 10) # + lien est porbable, plus c'est large
        
        # On ajoute le lien avec un titre (tooltip au survol)
        hover_text = f"Dominance: {row['Dominance_Index']:.2f} | Proba: {row['proba']:.2f}"
        net.add_edge(u, v, color=color_hex, title=hover_text, width=edge_width)

    net.force_atlas_2based() # Un algorithme qui fait bien ressortir les clusters
    return net.show(filename)

def plot_dominance_distribution(explainability_dataset, title="Distribution de la Dominance (Spatial vs SBM)"):
    """
    Affiche l'histogramme et la densité du Dominance_Index. -1 = 100% Spatial | 1 = 100% SBM
    """
    plt.figure(figsize=(10, 6))
    
    sns.histplot(explainability_dataset['Dominance_Index'], kde=True, color='purple', bins=30)
    plt.axvline(0, color='red', linestyle='--', alpha=0.6, label='Équilibre')
    
    plt.text(-0.9, plt.gca().get_ylim()[1]*0.9, '← Dominance SPATIAL', color='blue', fontweight='bold')
    plt.text(0.4, plt.gca().get_ylim()[1]*0.9, 'Dominance SBM →', color='darkred', fontweight='bold')
    
    plt.title(title, fontsize=14)
    plt.xlabel('Dominance Index', fontsize=12)
    plt.ylabel('Nombre de liens (Vrais Positifs)', fontsize=12)
    plt.xlim(-1.1, 1.1)  # On force les limites de l'index
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    
    plt.show()


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
        dataset_hidden = load_dataset(filename=f"dataset_hidden_{G_name}", talk = talk)
    except Exception:
        print(f"Dataset d'Évaluation introuvable pour {G_name}.")
        dataset_hidden = None

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

    return G_train, dataset_train, dataset_hidden, xgboost_data, shap_explainer, shap_analysis

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