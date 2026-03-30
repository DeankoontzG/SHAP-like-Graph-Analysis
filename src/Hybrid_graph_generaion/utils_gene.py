import pandas as pd
import numpy as np
import networkx as nx
import graph_tool.all as gt
from graph_tool.spectral import adjacency
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import fsolve, minimize_scalar
from scipy.stats import ks_2samp 


#######################################
###### FONCTIONS POUR GENERAZAO #######
#######################################

def get_real_graph_properties_sbm_V2(G_train):
    nodes_map = {node: i for i, node in enumerate(G_train.nodes())}
    edges = [(nodes_map[u], nodes_map[v]) for u, v in G_train.edges()]    
    communities = nx.get_node_attributes(G_train, 'sbm_id')
    unique_comms = sorted(list(set(communities.values())))
    mapping = {raw_id: i for i, raw_id in enumerate(unique_comms)}
    
    g = gt.Graph(directed=False)
    g.add_vertex(len(G_train.nodes()))
    g.add_edge_list(edges)
    
    b_array = np.array([mapping[communities[node]] for node in G_train.nodes()])
    b_prop = g.new_vertex_property("int", b_array)

    state = gt.BlockState(g, b=b_prop, deg_corr=True)

    # La matrice de liens entre blocs
    e_rs = state.get_matrix().toarray()
    
    # Les degrés de chaque nœud
    k = g.get_out_degrees(g.get_vertices())
    
    return e_rs, k, b_array

def get_real_graph_properties_pos_V2(G_train, n_components=4, shuffle=True):
    nodes = list(G_train.nodes())
    embeddings_attr = nx.get_node_attributes(G_train, 'deepwalk')
    raw_embeddings = np.array([embeddings_attr[node] for node in nodes])
    degrees = []
    
    for node in nodes:
        degrees.append(G_train.degree(node))
    
    # On réduit les dimensions des embeddings pour éviter des distances trop grandes. Utile ?
    pca = PCA(n_components=n_components, random_state=42)
    pos_reduced = pca.fit_transform(raw_embeddings)
    
    # Normalisation dans [0;1], standard pour modèles spatiaux askip
    pos_min = pos_reduced.min(axis=0)
    pos_max = pos_reduced.max(axis=0)
    pos_normalized = (pos_reduced - pos_min) / (pos_max - pos_min)
        
    if shuffle:
        rng_pos = np.random.default_rng(seed=42)
        idx_pos = rng_pos.permutation(len(pos_normalized))
        pos_final = pos_normalized[idx_pos]
        
        rng_deg = np.random.default_rng(seed=99) 
        idx_deg = rng_deg.permutation(len(degrees))
        degrees_final = np.array(degrees)[idx_deg]
    else:
        degrees_final = pos_normalized
        
    return degrees_final, pos_final

def get_probs_sbm_DC(e_rs, k, b):
    e_r = e_rs.sum(axis=1)
    n = len(k)
    n_blocks = e_rs.shape[0]
    P = np.zeros((n, n))
    
    for r in range(n_blocks):
        for s in range(r, n_blocks):
            idx_r = np.where(b == r)[0]
            idx_s = np.where(b == s)[0]
            if e_r[r] == 0 or e_r[s] == 0: continue
            
            # Formule de Karrer & Newman
            block_p = np.outer(k[idx_r], k[idx_s]) * (e_rs[r, s] / (e_r[r] * e_r[s]))
            
            P[np.ix_(idx_r, idx_s)] = block_p
            if r != s:
                P[np.ix_(idx_s, idx_r)] = block_p.T

    np.fill_diagonal(P, 0)

    n_clipped = np.sum(P[np.triu_indices(n, k=1)] > 1.0)
    if n_clipped > 0:
        print(f"Warning: {int(n_clipped)} probabilités SBM plafonnées à 1.0 (Hubs importants)")
    P = np.clip(P, 0, 1)
    
    return P

def get_probs_spatial_DC(degrees, positions, sigma=1.0, iterations=4000, lr=0.2):
    n = len(degrees)
    k = np.array(degrees, dtype=float)
    dist_matrix = squareform(pdist(positions, 'euclidean'))
    
    alpha = np.log(k + 1e-5)
    deterrence_fct = sigma * dist_matrix
    
    for _ in range(iterations):
        # Logit = alpha_i + alpha_j - sigma * dist_ij
        logit = alpha[:, np.newaxis] + alpha[np.newaxis, :] - deterrence_fct
        # Pij = ni*nj*deterrence_fct/ (1 + ni*nj*deterrence_fct) => Alternative sans multi-arrêtes, P in [O;1]
        P = 1.0 / (1.0 + np.exp(-logit))
        np.fill_diagonal(P, 0)
        
        # 2. Erreur sur les degrés
        current_degs = P.sum(axis=1)
        diff = k - current_degs
        
        # 3. Mise à jour des potentiels (Descente de gradient)
        alpha += lr * diff / n 

    n_sattures = (P > 0.99).sum()
    print(f"Warning: {n_sattures} probabilités saturées à +0.99")
    mae_deg = np.mean(np.abs(P.sum(axis=1) - degrees))
    print(f"Erreur moyenne sur les degrés (MAE) : {mae_deg:.6f}")
        
    return P

def get_probs_sbm_non_DC(e_rs, b):
    n = len(b)
    n_blocks = e_rs.shape[0]
    P = np.zeros((n, n))
    
    counts = np.bincount(b)
    
    for r in range(n_blocks):
        for s in range(r, n_blocks):
            idx_r = np.where(b == r)[0]
            idx_s = np.where(b == s)[0]
            
            # Calcul du nombre de liens maximum possibles entre ces blocs
            if r == s:
                possible = counts[r] * (counts[r] - 1) / 2
                p_rs = e_rs[r, s] / (2*possible)
            else:
                possible = counts[r] * counts[s]
                p_rs = e_rs[r, s] / possible
                
            P[np.ix_(idx_r, idx_s)] = p_rs
            if r != s:
                P[np.ix_(idx_s, idx_r)] = p_rs
                
    np.fill_diagonal(P, 0)
    n_clipped = np.sum(P > 1.0)
    if n_clipped > 0:
        print(f"Warning: {int(n_clipped/2)} probabilités spatiales plafonnées à 1.0")
    P = np.clip(P, 0, 1)
    
    return P

def get_probs_spatial_non_DC(positions, n_liens_target, sigma=1.0):
    n = len(positions)
    dist_matrix = squareform(pdist(positions, 'euclidean'))

    deterrence = sigma * dist_matrix
    iu = np.triu_indices(n, k=1)
    det_vec = deterrence[iu]

    def objective(alpha):
        logits = alpha - det_vec
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.sum(probs) - n_liens_target

    alpha_opt = fsolve(objective, x0=0.0)[0]
    
    logit_final = alpha_opt - deterrence
    P = 1.0 / (1.0 + np.exp(-logit_final))
    np.fill_diagonal(P, 0)
    
    print(f" Alpha trouvé : {alpha_opt:.4f} pour {n_liens_target} liens visés.")
    return P


def generate_graph_from_probs(P, sbm_groups=None, positions=None):
    n = P.shape[0]
    g = gt.Graph(directed=False)
    g.add_vertex(n)
    
    upper_idx = np.triu_indices(n, k=1)
    probs_vector = P[upper_idx]
    
    mask = np.random.random(len(probs_vector)) < probs_vector
    edges = np.column_stack((upper_idx[0][mask], upper_idx[1][mask]))
    
    g.add_edge_list(edges)
        
    return g

def generate_graph_benchmarks(Hybrid_ratios_list, P_sbm, P_spatial, position, commu, name="00_OUBLI_DE_NOM"):
    results_list = []

    for alpha in Hybrid_ratios_list:
        G_name = f"{name}_{f'{alpha:.2f}'.replace('.', '_')}_pos_{f'{1-alpha:.2f}'.replace('.', '_')}.graphml"

        print("\n" + "="*90)
        print(f"Pour ratio_sbm = {alpha}")
        print("\n" + "="*90)
        
        P_hybride = P_sbm * alpha + P_spatial * (1 - alpha)
        g_hybride = generate_graph_from_probs(P_hybride)

        g_hybride_nx = convert_to_nx_with_metadata(g_hybride, position, commu)
        save_as_graphml(g_hybride_nx, filename=G_name)

        node_0_data = g_hybride_nx.nodes[0]
        print(f"ID SBM : {node_0_data['GT_sbm_id']} (Type: {type(node_0_data['GT_sbm_id'])})")
        print(f"Position : {node_0_data['GT_pos']}")
        
        var_h = get_variance_from_P(P_hybride)
        ent_h = get_entropy_from_p(P_hybride)
        ll_h = get_log_likelihood(g_hybride, P_hybride)
        
        clustering = gt.global_clustering(g_hybride)[0]  
        
        results_list.append({
            "Modèle": f"Hybride (α={alpha:.2f})",
            "N": g_hybride.num_vertices(),
            "E": g_hybride.num_edges(),
            "Variance": f"{var_h:.8f}",
            "Entropy": f"{ent_h:.2f}",
            "Log-Likelihood": f"{ll_h:.2f}",
            "Clustering": f"{clustering:.4f}"
        })

    # --- Affichage final ---
    df_results = pd.DataFrame(results_list)

    print("\n" + "="*90)
    print("📊 TABLEAU RÉCAPITULATIF DE L'HYBRIDATION")
    print("="*90)
    print(df_results)
    print("="*90)


#####################################
###### FONCTIONS POUR ANALYSE #######
#####################################

def get_variance_from_P(P):
    n = P.shape[0]
    upper_idx = np.triu_indices(n, k=1)
    
    p_vector = P[upper_idx].copy()
    variance = np.var(p_vector)
    
    return variance

def plot_degree_correlation(degrees_orig, g_sbm, g_spatial):
    degs_sbm = [v.out_degree() for v in g_sbm.vertices()]
    degs_spatial = [v.out_degree() for v in g_spatial.vertices()]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    # --- Sous-graphe 1 : SBM ---
    ax1.scatter(degrees_orig, degs_sbm, alpha=0.5, color='royalblue', label='Noeuds')
    ax1.plot([0, max(degrees_orig)], [0, max(degrees_orig)], 'r--', label='Identité (Parfait)')
    ax1.set_title("Fidélité des degrés : SBM Pure")
    ax1.set_xlabel("Degré Original (G_train)")
    ax1.set_ylabel("Degré Généré")
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()
    
    # --- Sous-graphe 2 : Spatial ---
    ax2.scatter(degrees_orig, degs_spatial, alpha=0.5, color='forestgreen', label='Noeuds')
    ax2.plot([0, max(degrees_orig)], [0, max(degrees_orig)], 'r--', label='Identité (Parfait)')
    ax2.set_title(f"Fidélité des degrés : Spatial Pure")
    ax2.set_xlabel("Degré Original (G_train)")
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # 3. Métrique d'erreur (MAE)
    mae_sbm = np.mean(np.abs(np.array(degrees_orig) - np.array(degs_sbm)))
    mae_spatial = np.mean(np.abs(np.array(degrees_orig) - np.array(degs_spatial)))
    
    print(f"Erreur Moyenne Absolue (MAE) SBM     : {mae_sbm:.2f}")
    print(f"Erreur Moyenne Absolue (MAE) Spatial : {mae_spatial:.2f}")

def plot_degree_distribution(degrees_orig, g_sbm, g_spatial, g_test, label_test="Hybride"):
    d_sbm, d_spat, d_test = [[v.out_degree() for v in g.vertices()] for g in [g_sbm, g_spatial, g_test]]
    fig, ax = plt.subplots(figsize=(10, 6))
    params = {'bw_adjust': 0.2, 'cut': 0, 'gridsize': 200}
    
    # Courbes de densité
    sns.kdeplot(degrees_orig, ax=ax, color='#7f8c8d', label='Original', fill=True, alpha=0.1, linewidth=1, linestyle='--', **params)
    sns.kdeplot(d_sbm, ax=ax, color='#5d6d7e', label='SBM Pur', linewidth=1.5, alpha=0.8, **params)
    sns.kdeplot(d_spat, ax=ax, color='#a93226', label='Spatial Pur', linewidth=1.5, alpha=0.5, linestyle=':', **params)
    sns.kdeplot(d_test, ax=ax, color='#d35400', label=label_test, linewidth=3, alpha=1.0, **params)

    # Cosmétique
    ax.set_title("Densités de Degrés : Impact de l'hybridation décorrélée", fontsize=12, pad=15)
    ax.set_xlabel("Degré (k)"); ax.set_ylabel("Densité"); ax.grid(True, linestyle=':', alpha=0.5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.legend(frameon=False)
    plt.tight_layout(); plt.show()

    # Stats KS
    print(f"KS Distances -> SBM: {ks_2samp(degrees_orig, d_sbm).statistic:.4f} | Spatial: {ks_2samp(degrees_orig, d_spat).statistic:.4f} | {label_test}: {ks_2samp(degrees_orig, d_test).statistic:.4f}")
    print(f"Moyennes -> Orig: {np.mean(degrees_orig):.2f} | {label_test}: {np.mean(d_test):.2f}")
    print(f"Std -> Orig: {np.std(degrees_orig):.2f} | {label_test}: {np.std(d_test):.2f}")

def get_entropy_from_p(P):
    upper_idx = np.triu_indices_from(P, k=1)
    p_vector = P[upper_idx]
    
    epsilon = 1e-12
    p_vector = np.clip(p_vector, epsilon, 1 - epsilon)
    
    # Formule de l'entropie binaire : H = - [p*log2(p) + (1-p)*log2(1-p)]
    h_binaire = -(p_vector * np.log2(p_vector) + (1 - p_vector) * np.log2(1 - p_vector))
    total_entropy = np.sum(h_binaire)
    
    return total_entropy

def get_log_likelihood(G, P):
    if isinstance(G, nx.Graph):
        adj = nx.to_numpy_array(G, nodelist=range(len(P)))
    else:
        adj = adjacency(G).toarray()

    upper_idx = np.triu_indices_from(P, k=1)
    p_vector = np.clip(P[upper_idx], 1e-12, 1 - 1e-12)
    adj_vector = adj[upper_idx]

    log_likelihood = np.sum(adj_vector * np.log2(p_vector) + (1 - adj_vector) * np.log2(1 - p_vector))
    
    return log_likelihood

def analyze_errors(G, P):
    adj = adjacency(G).toarray()
    upper = np.triu_indices_from(P, k=1)
    p_vec = P[upper]
    a_vec = adj[upper]
    
    surprises = p_vec[a_vec == 1]
    print(f"Proba moyenne pour les arêtes réelles : {np.mean(surprises):.6f}")
    print(f"Proba min pour une arête réelle : {np.min(surprises):.12f}")
    
    critiques = np.sum(surprises < 1e-4)
    print(f"Nombre d'arêtes 'impossibles' selon le modèle : {critiques}")

def convert_to_nx_with_metadata(gt_graph, positions, sbm_labels):
    edges = gt_graph.get_edges()
    n_nodes = len(sbm_labels)
    G_nx = nx.Graph()
    G_nx.add_nodes_from(range(n_nodes))
    G_nx.add_edges_from(edges)
    
    sbm_dict = {i: int(sbm_labels[i]) for i in range(n_nodes)}
    pos_dict = {i: str(list(positions[i])) for i in range(n_nodes)}

    nx.set_node_attributes(G_nx, sbm_dict, "GT_sbm_id")
    nx.set_node_attributes(G_nx, pos_dict, "GT_pos")
    
    return G_nx

def save_as_graphml(G_nx, filename="mon_graphe.graphml", folder="../../graph_library"):
    path = os.path.join(folder, filename)
    nx.write_graphml(G_nx, path)
    print(f"Graphe exporté avec succès dans : {path}")

def visualize_generation_diagnostics_v2(P_sbm, P_spatial, k_original, e_rs, b_groups):
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    
    # --- 1. Probabilités P_ij ---
    n = P_sbm.shape[0]
    iu = np.triu_indices(n, k=1)
    sns.kdeplot(P_sbm[iu][P_sbm[iu] > 1e-4], ax=axes[0], label="SBM", fill=True, color="blue")
    sns.kdeplot(P_spatial[iu][P_spatial[iu] > 1e-4], ax=axes[0], label="Spatial", fill=True, color="orange")
    axes[0].set_title("Répartition des Probabilités $P_{ij}$")
    axes[0].set_xlim(0, 1)
    axes[0].legend()

    # --- 2. Degrés ---
    sns.histplot(k_original, ax=axes[1], kde=True, color='teal')
    axes[1].set_title(f"Degrés (Moy: {np.mean(k_original):.1f}, STD: {np.std(k_original):.1f})")

    # --- 3. Matrice de Densité + Tailles des Communautés ---
    counts = np.bincount(b_groups)
    n_blocks = len(counts)
    block_prob_matrix = np.zeros((n_blocks, n_blocks))
    
    for r in range(n_blocks):
        for s in range(r, n_blocks):
            possible = counts[r] * (counts[r] - 1) / 2 if r == s else counts[r] * counts[s]
            if possible > 0:
                val = e_rs[r, s] / possible
                block_prob_matrix[r, s] = block_prob_matrix[s, r] = val

    # Affichage de la Heatmap
    im = sns.heatmap(block_prob_matrix, ax=axes[2], cmap="YlOrRd", vmin=0, vmax=1, annot=False)
    
    # AJOUT : Tailles des communautés sur les axes
    axes[2].set_xticklabels([f"ID:{i}\n(n={counts[i]})" for i in range(n_blocks)], rotation=45)
    axes[2].set_yticklabels([f"ID:{i} (n={counts[i]})" for i in range(n_blocks)], rotation=0)
    
    axes[2].set_title("Densité SBM et Taille des Blocs")
    
    plt.tight_layout()
    plt.show()

def match_spatial_to_sbm_variance(P_sbm, degrees, positions, DC=True):
    target_variance = get_variance_from_P(P_sbm)
    target_links = np.sum(degrees) / 2
    print(f"Variance cible (SBM) : {target_variance:.8f}")
    print("-" * 30)

    history = {'step': 0}

    def objective(sigma_test):
        history['step'] += 1
        
        if DC : 
            P_test = get_probs_spatial_DC(degrees, positions, sigma=sigma_test)
        else : 
            P_test = get_probs_spatial_non_DC(positions, n_liens_target= target_links,sigma=sigma_test)
        current_var = get_variance_from_P(P_test)
        diff = abs(current_var - target_variance)
        
        print(f"Step {history['step']:02d} | Sigma testé: {sigma_test:.4f} | Var: {current_var:.8f} | Δ: {diff:.2e}")
        
        return (current_var - target_variance)**2

    res = minimize_scalar(objective, bounds=(0.005, 50), method='bounded')
    
    print("-" * 30)
    opt_sigma = res.x
    
    if DC : 
        final_P_spatial = get_probs_spatial_DC(degrees, positions, sigma=opt_sigma)
    else : 
        final_P_spatial = get_probs_spatial_non_DC(positions, n_liens_target= target_links, sigma=opt_sigma)
    final_var = get_variance_from_P(final_P_spatial)
    
    print(f"✨ Sigma optimal trouvé : {opt_sigma:.4f}")
    print(f"📊 Variance finale Spatial : {final_var:.8f} (Écart: {abs(final_var-target_variance):.2e})")
    
    return opt_sigma, final_P_spatial