import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist

from .MetaLouvain import best_partition, partition_at_level 

def compute_optimal_beta(A, K, M):
    """
    Calcule analytiquement le beta optimal par projection des moindres carrés
    pour minimiser || A - ((1-beta)K + beta M) ||_F^2
    """
    num = np.sum((A - K) * (M - K))
    den = np.sum((M - K) ** 2)
    
    if den == 0:
        return 0.0
    
    # Sécurité mathématique : beta doit rester dans [0, 1]
    return float(np.clip(num / den, 0.0, 1.0))

def compute_dc_sbm_matrix(A, degrees, partition_dict):
    """
    Calcule la matrice M du modèle nul DC-SBM (Karrer-Newman)
    """
    n = A.shape[0]
    nodes = list(range(n)) # On suppose des indices contigus 0 à N-1
    
    # Extraction et alignement des blocs
    z = np.array([partition_dict[i] for i in nodes])
    num_blocks = len(np.unique(z))
    
    # 1. Volume de degrés par communauté (kappa_r)
    kappa = np.bincount(z, weights=degrees, minlength=num_blocks)
    kappa = np.where(kappa == 0, 1e-10, kappa) # Sécurité division
    
    # 2. Matrice d'interopérabilité des blocs (m_rs)
    m_rs = np.zeros((num_blocks, num_blocks))
    for r in range(num_blocks):
        for s in range(num_blocks):
            mask_r = (z == r)
            mask_s = (z == s)
            m_rs[r, s] = np.sum(A[mask_r][:, mask_s])
            
    # 3. Reconstruction de M_ij = (k_i / kappa_zi) * (k_j / kappa_zj) * m_{zi, zj}
    theta = degrees / kappa[z]
    M = np.outer(theta, theta) * m_rs[z[:, None], z]
    np.fill_diagonal(M, 0)
    
    return M

def compute_dc_gravity_matrix(degrees, dist_matrix, max_iter=100, tol=1e-4):
    """
    Calcule M' (modèle gravitaire degree-corrected) via l'algorithme de Sinkhorn (RAS)
    f(d_ij) = exp(-d_ij) ou d_ij^-2. On utilise ici une décroissance exponentielle.
    """
    n = len(degrees)
    # Fonction de friction spatiale
    F = np.exp(-1.0 * dist_matrix)
    np.fill_diagonal(F, 0)
    
    # Initialisation des balances/multiplicateurs de Lagrange (u_i)
    u = degrees.copy()
    
    for _ in range(max_iter):
        u_old = u.copy()
        denom = np.dot(F, u)
        denom = np.where(denom == 0, 1e-10, denom)
        u = degrees / denom
        
        if np.mean(np.abs(u - u_old)) < tol:
            break
            
    M_prime = np.outer(u, u) * F
    return M_prime

def compute_matrix_rv_coefficient(M, M_prime):
    """
    Calcule le coefficient RV (corrélation matricielle) entre le modèle nul SBM (M)
    et le modèle nul spatial (M').
    Varie entre 0 (orthogonalité parfaite) et 1 (redondance totale).
    """
    # On s'assure d'enlever les diagonales si ce n'est pas déjà fait
    # pour ne pas polluer avec des zéros ou des autocorrélations
    num = np.sum(M * M_prime)
    den = np.sqrt(np.sum(M ** 2) * np.sum(M_prime ** 2))
    
    if den == 0:
        return 0.0
        
    return float(num / den)

def compute_embedding_positive_only(R, embedding_dim=64):
    """
    Approche de l'article : Ne conserve que les k plus grandes valeurs propres POSITIVES.
    Modèle sous-jacent : R ≈ X @ X.T (Espace euclidien pur / Assortatif)
    """
    n = R.shape[0]
    
    # 1. Diagonalisation propre de la matrice réelle symétrique
    vals, vecs = np.linalg.eigh(R)
    
    # 2. Filtrage : On ne garde que les indices où les valeurs propres sont strictement positives
    pos_mask = vals > 0
    vals_pos = vals[pos_mask]
    vecs_pos = vecs[:, pos_mask]
    
    # 3. Tri par ordre décroissant des valeurs positives
    idx_sorted = np.argsort(vals_pos)[::-1]
    
    # Sécurité : on prend le minimum entre la dimension demandée et le nombre de valeurs positives disponibles
    actual_dim = min(embedding_dim, len(idx_sorted))
    idx_top = idx_sorted[:actual_dim]
    
    # 4. Extraction des composantes dominantes
    Lambda_d = vals_pos[idx_top]
    V_d = vecs_pos[:, idx_top]
    
    # 5. Construction de l'embedding (Racine carrée directe car Lambda > 0)
    X_embedding = V_d * np.sqrt(Lambda_d)
    
    print(f" > Embedding Positif : {actual_dim} dimensions extraites (sur {embedding_dim} demandées).")
    return X_embedding

def compute_embedding_absolute_magnitude(R, embedding_dim=64):
    """
    Approche Eckart-Young : Conserve les k plus grandes valeurs propres en VALEUR ABSOLUE.
    Modèle sous-jacent : R ≈ X @ S @ X.T (Espace indéfini / Capture Attraction et Répulsion)
    """
    n = R.shape[0]
    
    # 1. Diagonalisation propre
    vals, vecs = np.linalg.eigh(R)
    
    # 2. Tri par la magnitude absolue (du plus grand au plus petit)
    idx_sorted = np.argsort(np.abs(vals))[::-1]
    
    # Sélection des d premières composantes
    actual_dim = min(embedding_dim, n - 2)
    idx_top = idx_sorted[:actual_dim]
    
    Lambda_d = vals[idx_top]
    V_d = vecs[:, idx_top]
    
    # 3. Construction de l'embedding réel via la valeur absolue de la magnitude
    X_embedding = V_d * np.sqrt(np.abs(Lambda_d))
    
    # OPTIONNEL : Si vous avez besoin de S pour reconstruire le résidu exact un jour :
    # S_diagonal = np.sign(Lambda_d)
    
    num_pos = np.sum(Lambda_d > 0)
    num_neg = np.sum(Lambda_d < 0)
    print(f" > Embedding Absolu : {actual_dim} dimensions extraites ({num_pos} positives, {num_neg} négatives).")
    
    return X_embedding

def run_decoupled_framework(G, max_global_iters=10, emb_method=PosEigenval):
    """
    Fonction parente itérative (Étapes 1, 2, 3)
    """
    print("=== Lancement du Framework d'Inférence Décorrélée ===")
    
    # Pré-calculs structurels stables
    nodes = list(G.nodes())
    n = len(nodes)
    mapping = {node: i for i, node in enumerate(nodes)}

    A = nx.to_numpy_array(G, nodelist=nodes)
    degrees = np.sum(A, axis=1)
    m2 = np.sum(degrees)
    
    if m2 == 0:
        raise ValueError("Le graphe ne contient aucune arête.")
        
    # Modèle nul de configuration de base (K)
    K = np.outer(degrees, degrees) / m2
    np.fill_diagonal(K, 0)
    
    # --- ÉTAPE 1 : Louvain Initial (A - K) ---
    print("\n[Étape 1] Initialisation de la structure de communauté via Louvain...")
    def initial_null_model(u, v):
        return K[mapping[u], mapping[v]]
        
    # On appelle votre fonction issue de MetaLouvain
    init_partition = best_partition(G, resolution=1.0, null_model=initial_null_model)
    
    # Traduction en indices matriciels [0, N-1]
    current_partition_dict = {mapping[node]: com for node, com in init_partition.items()}
    
    M = compute_dc_sbm_matrix(A, degrees, current_partition_dict)
    
    # --- BOUCLE PRINCIPALE (Itérations des Étapes 2 et 3) ---
    for idx_iter in range(1, max_global_iters + 1):
        print(f"\n--- Global Iteration #{idx_iter} ---")
        
        # --- ÉTAPE 2 : Ajustement du SBM & Calcul de l'Embedding ---
        beta_M = compute_optimal_beta(A, K, M)
        print(f" > Beta_M optimal (SBM) calculé : {beta_M:.4f} (Poids de K: {1-beta_M:.4f})")
        
        # Matrice résiduelle pour l'embedding
        R_embedding = A - ((1 - beta_M) * K + beta_M * M)
        
        # Calcul de l'embedding associé, via vecteurs propres
        embedding_dim = min(64, n - 2)
        if emb_method =="PosEigenvals":
            X_embedding = compute_embedding_positive_only(R_embedding, embedding_dim)
        elif emb_method == "AllEigenvals" : 
            X_embedding = compute_embedding_absolute_magnitude(R_embedding, embedding_dim)
        else : 
            print(f"U cappin' bro, emb_method {emb_method} is pure shite.")
        
        # --- ÉTAPE 3 : Modèle Gravitaire & Re-détection de Communautés ---
        # Matrice des distances géométriques induites dans l'espace abstrait
        dist_matrix = cdist(X_embedding, X_embedding, metric='euclidean')
        
        M_prime = compute_dc_gravity_matrix(degrees, dist_matrix)
        beta_M_prime = compute_optimal_beta(A, K, M_prime)
        print(f" > Beta_M' optimal (Spatial) calculé : {beta_M_prime:.4f}")
        
        # Matrice de fond globale combinée pour le prochain Louvain
        W_null = (1 - beta_M_prime) * K + beta_M_prime * M_prime
        
        # Encapsulation matricielle pour votre méthode Louvain Custom
        def iterative_null_model(u, v):
            return W_null[mapping[u], mapping[v]]
            
        # Exécution du Louvain épuré de l'information géométrique/spatiale
        new_partition = best_partition(G, resolution=1.0, null_model=iterative_null_model)
        current_partition_dict = {mapping[node]: com for node, com in new_partition.items()}

        # --- MISE À JOUR DU SBM AVEC LA PARTITION TOUT JUSTE INFÉRÉE ---
        # Maintenant, M et M' sont tous les deux basés sur l'itération courante
        M = compute_dc_sbm_matrix(A, degrees, current_partition_dict)
        
        # --- MESURE DE CONVERGENCE STRUCTURELLE ---
        current_rv = compute_matrix_rv_coefficient(M, M_prime)
        print(f" > Alignement SBM vs Spatial (Coefficient RV) : {current_rv:.4f}")

        last_rv = current_rv
                    
    # Reconstruction finale du dictionnaire au format NetworkX d'origine
    final_partition_nx = {nodes[i]: com for i, com in current_partition_dict.items()}    
    
    return final_partition_nx, X_embedding