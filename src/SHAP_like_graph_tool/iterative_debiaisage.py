from .MetaLouvain import best_partition, partition_at_level 

import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from scipy.spatial import procrustes
from scipy.optimize import minimize_scalar
from sklearn.metrics import adjusted_rand_score
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif


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

def compute_optimal_beta_orthogonal(A, K, Model_Current, Model_Other, lambda_ortho=10.0):
    """
    Calcule beta pour minimiser || A - ((1-beta)K + beta Model_Current) ||_F^2
    PÉNALISÉ par l'alignement RV avec l'autre modèle (Model_Other).
    """
    if Model_Other is None or np.sum(Model_Other) == 0:
        num = np.sum((A - K) * (Model_Current - K))
        den = np.sum((Model_Current - K) ** 2)
        return 0.0 if den == 0 else float(np.clip(num / den, 0.0, 1.0))

    def loss_function(beta):
        W_null = (1 - beta) * K + beta * Model_Current
        reconstruction_error = np.sum((A - W_null) ** 2)
        
        num_rv = np.sum(W_null * Model_Other)
        den_rv = np.sqrt(np.sum(W_null ** 2) * np.sum(Model_Other ** 2))
        rv_coeff = 0.0 if den_rv == 0 else num_rv / den_rv
        
        return reconstruction_error + lambda_ortho * (rv_coeff ** 2)

    res = minimize_scalar(loss_function, bounds=(0.0, 1.0), method='bounded')
    return float(res.x)


def compute_mutual_information_mixed(X_emb, partition_dict):
    """
    Calcule la MI moyenne entre un embedding continu (X_emb) 
    et une partition de communautés discrètes (partition_dict).
    """
    n = X_emb.shape[0]
    # On aligne les labels de communauté dans l'ordre des indices matriciels [0, N-1]
    labels = np.array([partition_dict[i] for i in range(n)])
    
    # Si Louvain n'a trouvé qu'une seule communauté (cas pathologique rare), la MI est nulle
    if len(np.unique(labels)) <= 1:
        return 0.0
        
    mi_scores = []
    num_dimensions = X_emb.shape[1]
    
    # Calcul de la MI pour chaque dimension de l'espace géométrique
    for i in range(num_dimensions):
        score = mutual_info_classif(X_emb[:, i].reshape(-1, 1), labels, random_state=42)[0]
        mi_scores.append(score)
        
    return float(np.mean(mi_scores))
    

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

def compute_dc_gravity_matrix_old(degrees, dist_matrix, max_iter=100, tol=1e-4):
    n = len(degrees)
    target_sum = np.sum(degrees) # C'est notre masse totale 2m
    
    med_dist = np.median(dist_matrix[dist_matrix > 0])
    if med_dist == 0:
        med_dist = 1.0
        
    # 1. Calcul de la friction (on laisse la diagonale à 1 pour la stabilité du processus)
    F = np.exp(-1.0 * (dist_matrix / med_dist))
    
    # 2. Boucle de Sinkhorn standardisée
    u = np.ones(n) # Initialisation uniforme plus stable que 'degrees'
    
    for _ in range(max_iter):
        u_old = u.copy()
        denom = np.dot(F, u)
        denom = np.where(denom < 1e-12, 1e-12, denom)
        u = degrees / denom
        if np.mean(np.abs(u - u_old)) < tol:
            break
            
    M_prime = np.outer(u, u) * F
    np.fill_diagonal(M_prime, 0)
    
    # 5. Redressement final de la masse (Mass Preservation Adjustment)
    # Pour s'assurer que les pertes dues à la diagonale nulle ou aux approximations n'altèrent pas la somme
    current_sum = np.sum(M_prime)
    if current_sum > 0:
        norm_ratio = (target_sum / current_sum)
        M_prime = M_prime * norm_ratio
        if norm_ratio > 2 or norm_ratio < 0.5 : 
            print(f"Masse totale de liens ajustée de {norm_ratio}")
        
    return M_prime

def compute_dc_gravity_matrix(degrees, dist_matrix, max_iter=1000, tol=1e-2):
    n = len(degrees)
    target_sum = np.sum(degrees)
    
    med_dist = np.median(dist_matrix[dist_matrix > 0])
    if med_dist == 0:
        med_dist = 1.0
        
    # 1. Friction avec un "bruit de fond" (régularisation epsilon)
    # On ajoute une valeur infime (1e-8) partout pour garantir que le support est total
    F = np.exp(-1.0 * (dist_matrix / med_dist)) + 1e-8
    np.fill_diagonal(F, 0.0) # On maintient la diagonale à 0
    
    # 2. Initialisation
    u = np.ones(n)
    alpha = 0.5 # Facteur d'amortissement pour stabiliser la convergence symétrique
    
    # 3. Boucle de Sinkhorn Symétrique Amortie
    for idx in range(max_iter):
        u_old = u.copy()
        
        Fu = np.dot(F, u)
        
        # Cible théorique pour l'itération actuelle
        u_target = np.sqrt(degrees / Fu)
        
        # Mise à jour amortie (évite les oscillations destructrices et le gel)
        u = alpha * u_old + (1 - alpha) * u_target
        
        # Le critère de convergence doit évaluer l'erreur par rapport aux degrés cibles,
        # et non la simple stagnation du vecteur u
        current_degrees = u * np.dot(F, u)
        error = np.max(np.abs(current_degrees - degrees))
        
        if error < tol:
            break
            
    # 4. Recomposition finale 
    M_prime = u[:, None] * F * u
    np.fill_diagonal(M_prime, 0.0)
    
    # Recalage de sécurité microscopique
    current_sum = np.sum(M_prime)
    if current_sum > 0:
        M_prime = M_prime * (target_sum / current_sum)
        
    print(f"-> SINKHORN SYMÉTRIQUE AMORTI : Itérations réelles = {idx+1}")
    print(f"-> Somme attendue (2m) : {target_sum:.2f} | Somme obtenue : {np.sum(M_prime):.2f}")
    
    return M_prime

def compute_nonDC_sbm_matrix(A, degrees, partition_dict):
    """
    Calcule la matrice M du modèle nul SBM Standard (Non-DC).
    M_ij dépend uniquement de la densité de liens entre la communauté de i et celle de j.
    """
    n = A.shape[0]
    nodes = list(range(n))
    z = np.array([partition_dict[i] for i in nodes])
    num_blocks = len(np.unique(z))
    
    block_sizes = np.bincount(z, minlength=num_blocks)
    
    B = np.zeros((n, num_blocks))
    B[np.arange(n), z] = 1
    
    m_rs = B.T @ A @ B
 
    block_sizes_safe = np.where(block_sizes == 0, 1, block_sizes)
    denom_sizes = np.outer(block_sizes_safe, block_sizes_safe)
    
    P_rs = m_rs / denom_sizes
    
    M = P_rs[z[:, None], z]
    np.fill_diagonal(M, 0.0)
    
    return M

def compute_nonDC_gravity_matrix(degrees, dist_matrix):
    """
    Calcule la matrice M' du modèle nul gravitaire Standard (Non-DC) 
    via projection de Poisson globale.
    """
    n = len(degrees)
    target_sum = np.sum(degrees) # Vaut 2m
    
    # 1. Calcul du rayon caractéristique de l'espace (Kernel Bandwidth)
    med_dist = np.median(dist_matrix[dist_matrix > 0])
    if med_dist == 0:
        med_dist = 1.0
        
    # 2. Matrice de friction pure (sans auto-boucles)
    F = np.exp(-1.0 * (dist_matrix / med_dist))
    np.fill_diagonal(F, 0.0)
    
    # 3. Résolution de l'intensité globale (Maximum de Vraisemblance de Poisson)
    # Dans un modèle homogène, l'intensité attendue par paire de nœuds (i,j) 
    # est directement liée à la densité de friction disponible.
    total_friction = np.sum(F)
    
    if total_friction == 0:
        return np.zeros_like(F)
        
    # Calcul de la constante d'interaction c 
    # Mathématiquement, c'est le taux qui maximise la vraisemblance du volume de liens
    c = target_sum / total_friction
    
    # Recomposition : l'intensité est distribuée de manière purement multiplicative
    M_prime = c * F
    
    print("-> GRAVITÉ STANDARD : Inférence par maximum de vraisemblance globale")
    print(f"-> Constante d'interaction (c) : {c:.6f}")
    print(f"-> Somme attendue (2m)         : {target_sum:.2f} | Somme obtenue : {np.sum(M_prime):.2f}")
    
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

def compute_orthogonal_procrustes_displacement(X_prev, X_curr):
    """
    Aligne X_curr sur X_prev via Procruste orthogonal (SVD) 
    et calcule la distance euclidienne réelle parcourue par chaque nœud.
    """
    # 1. Centrage des deux embeddings pour éliminer les translations
    X_prev_centered = X_prev - np.mean(X_prev, axis=0)
    X_curr_centered = X_curr - np.mean(X_curr, axis=0)
    
    # 2. Résolution du problème de Procruste orthogonal (Trouver la rotation R)
    # Matrice de covariance croisée
    M = X_prev_centered.T @ X_curr_centered
    U, _, Vt = np.linalg.svd(M)
    R = Vt.T @ U.T  # Matrice de rotation optimale
    
    # 3. Projection de l'embedding courant dans le référentiel du précédent
    X_curr_aligned = X_curr_centered @ R
    
    # 4. Calcul des distances de déplacement par nœud
    # Vecteur de taille (N,) contenant la distance parcourue par chaque nœud
    displacements = np.linalg.norm(X_prev_centered - X_curr_aligned, axis=1)
    
    mean_disp = float(np.mean(displacements))
    max_disp = float(np.max(displacements))
    
    return mean_disp, max_disp
    
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


def run_decoupled_framework(G, max_global_iters=2, emb_method="PosEigenvals", DC = False, talk = False):
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
        
    init_partition = best_partition(G, resolution=1.0, null_model=initial_null_model)
    current_partition_dict = {mapping[node]: com for node, com in init_partition.items()}
    
    M = compute_dc_sbm_matrix(A, degrees, current_partition_dict)
    
     # --- LOGS SBM ---
    print(f"DEBUG SBM init - Somme M: {np.sum(M):.2f} (Attendu ≈ {np.sum(degrees):.2f})")
    print(f"DEBUG SBM init - Max M: {np.max(M):.4f} | Min M: {np.min(M):.4f}")
    print(f"DEBUG SBM init - Nb Communautés distinctes détectées: {len(np.unique(list(current_partition_dict.values())))}")

    X_embedding_old = None   
    W_null_old = K.copy()
    current_partition_old = None
    M_prime = None

    # --- BOUCLE PRINCIPALE (Itérations des Étapes 2 et 3) ---
    for idx_iter in range(1, max_global_iters + 1):
        print(f"\n--- Global Iteration #{idx_iter} ---")
        
        # --- ÉTAPE 2 : Ajustement du SBM & Calcul de l'Embedding ---
        beta_M = compute_optimal_beta(A, K, M)
        #beta_M = compute_optimal_beta(A, K, M, M_prime, lambda_ortho=50.0)
        print(f" > Beta_M optimal (SBM) calculé : {beta_M:.4f} (Poids de K: {1-beta_M:.4f})")
        
        # Matrice résiduelle pour l'embedding
        R_embedding = A - ((1 - beta_M) * K + beta_M * M)
        #R_embedding = A - 0.5 * (K + M)
        
        # Calcul de l'embedding associé, via vecteurs propres
        embedding_dim = min(8, n - 2)
        if emb_method =="PosEigenvals":
            X_embedding = compute_embedding_positive_only(R_embedding, embedding_dim)
        elif emb_method == "AllEigenvals" : 
            X_embedding = compute_embedding_absolute_magnitude(R_embedding, embedding_dim)
        else : 
            print(f"U cappin' bro, emb_method {emb_method} is pure shite.")

        if idx_iter ==1 : 
            X_embedding_one = X_embedding

        # --- ÉTAPE 3 : Modèle Gravitaire & Re-détection de Communautés ---
        dist_matrix = cdist(X_embedding, X_embedding, metric='euclidean')

        if DC : 
            M_prime = compute_dc_gravity_matrix(degrees, dist_matrix)
        else : 
            M_prime = compute_nonDC_gravity_matrix(degrees, dist_matrix)
            
        beta_M_prime = compute_optimal_beta(A, K, M_prime)
        print(f" > Beta_M' optimal (Spatial) calculé : {beta_M_prime:.4f}")

        W_null_candidate = (1 - beta_M_prime) * K + beta_M_prime * M_prime
        gamma_momentum = 0
        W_null = gamma_momentum * W_null_old + (1 - gamma_momentum) * W_null_candidate
        W_null_old = W_null.copy()  # Sauvegarde pour le tour d'après
        
        #W_null = 0.5 * (K + M_prime)
        
        # Encapsulation matricielle pour votre méthode Louvain Custom
        def iterative_null_model(u, v):
            return W_null[mapping[u], mapping[v]]
            
        # Exécution du Louvain épuré de l'information géométrique/spatiale. Initialisé à commus précédentes.
        new_partition = best_partition(G, resolution=1.0, null_model=iterative_null_model)

        """
        else:
            nodes_ordered = list(G.nodes())
            initial_partition = {nodes_ordered[i]: current_partition_dict[i] for i in range(n)}
            new_partition = best_partition(G, resolution=1.0, null_model=iterative_null_model, partition=initial_partition)
        """
            
        current_partition_dict = {mapping[node]: com for node, com in new_partition.items()}
        if idx_iter ==1 : 
            current_partition_one = {nodes[i]: com for i, com in current_partition_dict.items()}  

        # --- MISE À JOUR DU SBM AVEC LA PARTITION TOUT JUSTE INFÉRÉE ---
        # Maintenant, M et M' sont tous les deux basés sur l'itération courante
        if DC : 
            M = compute_dc_sbm_matrix(A, degrees, current_partition_dict)
        else : 
            M = compute_nonDC_sbm_matrix(A, degrees, current_partition_dict)

        if talk : 
            # --- LOGS EMBEDDING ---
            print(f"DEBUG EMBEDDING - Max coord X: {np.max(X_embedding):.4f} | Min coord X: {np.min(X_embedding):.4f}")
            print(f"DEBUG EMBEDDING - Somme des carrés (Norme Frobenius X): {np.sum(X_embedding**2):.2f}")
            print(f"DEBUG DISTANCES - Max dist: {np.max(dist_matrix):.4f} | Min dist (hors diag): {np.min(dist_matrix[dist_matrix > 0]):.4f} | Médiane: {np.median(dist_matrix[dist_matrix > 0]):.4f}")
    
            # --- LOGS SPATIAL & PROJECTION ---
            print(f"DEBUG SPATIAL - Max M': {np.max(M_prime):.4f} | Min M': {np.min(M_prime):.4f} | Somme M': {np.sum(M_prime):.2f}")
            # Calcul détaillé des termes du coefficient Beta pour identifier le blocage
            num_beta_prime = np.sum((A - K) * (M_prime - K))
            den_beta_prime = np.sum((M_prime - K) ** 2)
            print(f"DEBUG PROJECTION BETA_M' - Numérateur (Covariance A-K et M'-K): {num_beta_prime:.6f} | Dénominateur (Variance M'-K): {den_beta_prime:.6f}")
    
             # --- LOGS SBM ---
            print(f"DEBUG SBM - Somme M: {np.sum(M):.2f} (Attendu ≈ {np.sum(degrees):.2f})")
            print(f"DEBUG SBM - Max M: {np.max(M):.4f} | Min M: {np.min(M):.4f}")
            print(f"DEBUG SBM - Nb Communautés distinctes détectées: {len(np.unique(list(current_partition_dict.values())))}")
        
    
            # --- LOGS MATRICE DE FOND ---
            print(f"DEBUG NULL MODEL GLOBAL - Max W_null: {np.max(W_null):.4f} | Min W_null: {np.min(W_null):.4f}")
            print(f"DEBUG RESIDU LOUVAIN - Max (A - W_null): {np.max(A - W_null):.4f} | Min (A - W_null): {np.min(A - W_null):.4f}")

        # ---------------------------------------------------------------------
        # >>> INJECTION DU TRACKER DE DÉPLACEMENT ICI <<<
        # ---------------------------------------------------------------------
        if X_embedding_old is not None:
            # Sécurité si la dimension de l'embedding change (normalement fixe à 64)
            if X_embedding.shape == X_embedding_old.shape:
                mean_d, max_d = compute_orthogonal_procrustes_displacement(X_embedding_old, X_embedding)
                print(f"--> DYNAMIQUE GÉOMÉTRIQUE - Déplacement moyen des nœuds : {mean_d:.6f}")
                print(f"--> DYNAMIQUE GÉOMÉTRIQUE - Déplacement maximal constaté  : {max_d:.6f}")
            else:
                print("--> DYNAMIQUE GÉOMÉTRIQUE - Impossible de comparer (dimensions différentes)")
        else:
            print("--> DYNAMIQUE GÉOMÉTRIQUE - Itération 1 : Création de l'espace initial.")

        # --- MESURE DE CONVERGENCE STRUCTURELLE ---
        current_rv = compute_matrix_rv_coefficient(M, M_prime)
        print(f" > Alignement SBM vs Spatial (Coefficient RV) : {current_rv:.4f}")
        current_mi = compute_mutual_information_mixed(X_embedding, current_partition_dict)
        print(f" > Information Mutuelle entre Blocs et Espace (MI) : {current_mi:.6f}")

        # ---------------------------------------------------------------------
        # >>> TRACKER DE STABILITÉ DES COMMUNAUTÉS (ARI) <<<
        # ---------------------------------------------------------------------
        if current_partition_old is not None:
            # On extrait les vecteurs de labels alignés sur le même ordre de nœuds
            labels_old = np.array([current_partition_old[i] for i in range(n)])
            labels_curr = np.array([current_partition_dict[i] for i in range(n)])
            
            stability_ari = adjusted_rand_score(labels_old, labels_curr)
            print(f" > Stabilité de la partition (ARI vs Iter Précédente) : {stability_ari:.4f}")
            
            # Si l'espace ET les communautés ne bougent plus du tout : arrêt précoce
            if stability_ari == 1.0 and (X_embedding_old is not None and mean_d < 1e-4):
                print(f"\n[CONVERGENCE PARFAITE] Point fixe absolu atteint à l'itération {idx_iter} !")
                break
        else:
            print(" > Stabilité de la partition : Itération 1 (Pas de comparaison possible).")

        current_partition_old = current_partition_dict.copy()
        # On sauvegarde l'embedding actuel pour le tour suivant
        X_embedding_old = X_embedding.copy()
                    
    # Reconstruction finale du dictionnaire au format NetworkX d'origine
    final_partition_nx = {nodes[i]: com for i, com in current_partition_dict.items()}  

    print("\n=== Framework alterné continu terminé ===")
    return final_partition_nx, X_embedding, current_partition_one, X_embedding_one




def compute_mutual_information_spaces(X1, X2):
    """
    Calcule l'Information Mutuelle (MI) globale moyenne entre deux espaces continus,
    en évaluant l'indépendance de leurs dimensions respectives.
    """
    d1 = X1.shape[1]
    d2 = X2.shape[1]
    mi_scores = []
    
    # Évaluation croisée des dimensions
    for i in range(d1):
        for j in range(d2):
            # Estimateur non paramétrique basé sur les k-NN (robuste au non-linéaire)
            score = mutual_info_regression(X1[:, i].reshape(-1, 1), X2[:, j], random_state=42)[0]
            mi_scores.append(score)
            
    return float(np.mean(mi_scores))


def run_decoupled_framework_2embedds(G, max_global_iters=10, emb_method="PosEigenvals"):
    """
    Framework d'Inférence Décorrélée Alterné par Balancement Spectral Continu à Deux Espaces.
    Initialisation : Louvain Standard -> M0 Initial (Non-DC)
    Boucle : M0 -> Espace Spatial (AllEig) -> M1 -> Espace Topo (PosEig) -> M0
    """
    print("=== Lancement du Framework d'Inférence Décorrélée à Deux Espaces ===")
    
    # 1. Pré-calculs structurels stables
    nodes = list(G.nodes())
    n = len(nodes)
    mapping = {node: i for i, node in enumerate(nodes)}

    # Sécurité binaire pure
    A = nx.to_numpy_array(G, nodelist=nodes, weight=None)
    degrees = np.sum(A, axis=1)
    m2 = np.sum(degrees)
    
    if m2 == 0:
        raise ValueError("Le graphe ne contient aucune arête.")
        
    # Modèle nul de configuration global (K)
    K = np.outer(degrees, degrees) / m2
    np.fill_diagonal(K, 0)
    
    # Stabilisation de la dimension
    embedding_dim = min(8, n - 2)
    
    # 2. Amorçage de l'Étape 1 : Louvain Classique pour instancier le premier M0
    print("\n[Amorçage] Initialisation de la structure via Louvain classique...")
    def initial_null_model(u, v):
        return K[mapping[u], mapping[v]]
        
    from .MetaLouvain import best_partition 
    init_partition = best_partition(G, resolution=1.0, null_model=initial_null_model)
    current_partition_dict = {mapping[node]: com for node, com in init_partition.items()}
    
    # Premier modèle nul communautaire (Standard Non-DC vectorisé)
    M0 = compute_nonDC_sbm_matrix(A, degrees, current_partition_dict)

    # Historiques pour le suivi géométrique
    last_dist_spatial = None
    last_dist_topo = None

    # 3. Boucle principale de balancement spectral continu
    for idx_iter in range(1, max_global_iters + 1):
        print(f"\n--- Global Iteration #{idx_iter} ---")
        
        # =====================================================================
        # SOUS-ÉTAPE A : INFÉRENCE DE L'ESPACE SPATIAL (M1)
        # =====================================================================
        beta_M0 = compute_optimal_beta(A, K, M0)
        
        # Résidu dynamique optimisé (Exit le 0.5 fixe)
        R_spatial = A - ((1 - beta_M0) * K + beta_M0 * M0)
        
        # Extraction de la variance totale (Espace latent géométrique)
        if emb_method == "PosEigenvals":
            X_spatial = compute_embedding_positive_only(R_spatial, embedding_dim)
        else:
            X_spatial = compute_embedding_absolute_magnitude(R_spatial, embedding_dim)
            
        dist_spatial = cdist(X_spatial, X_spatial, metric='euclidean')
        
        # Suivi géométrique de l'Espace Spatial
        if last_dist_spatial is not None:
            corr_spat = np.corrcoef(dist_spatial.flatten(), last_dist_spatial.flatten())[0, 1]
            print(f" > Stabilité Espace Spatial (T vs T-1)     : {corr_spat:.6f}")
        last_dist_spatial = dist_spatial.copy()
        
        # Calcul du modèle nul spatial M1 Standard (Non-DC)
        M1_raw = compute_nonDC_gravity_matrix(degrees, dist_spatial)
        
        # --- PROJECTION ORTHOGONALE DE M1 PAR RAPPORT À M0 ---
        M0_centered = M0 - K
        M1_centered = M1_raw - K
        denom_project_1 = np.sum(M0_centered ** 2)
        if denom_project_1 > 1e-12:
            proj_factor_1 = np.sum(M0_centered * M1_centered) / denom_project_1
            M1_centered_ortho = M1_centered - (proj_factor_1 * M0_centered)
        else:
            M1_centered_ortho = M1_centered
            
        M1 = np.clip(K + M1_centered_ortho, 0.0, None)
        if np.sum(M1) > 0: M1 = M1 * (m2 / np.sum(M1))
        
        # =====================================================================
        # SOUS-ÉTAPE B : INFÉRENCE DE L'ESPACE TOPOLOGIQUE CONTINU (M0)
        # =====================================================================
        beta_M1 = compute_optimal_beta(A, K, M1)
        
        # Résidu nettoyé du signal spatial
        R_topo = A - ((1 - beta_M1) * K + beta_M1 * M1)
        
        # Extraction de l'homophilie pure (Communautés continues)
        X_topo = compute_embedding_positive_only(R_topo, embedding_dim)
        dist_topo = cdist(X_topo, X_topo, metric='euclidean')
        
        # Suivi géométrique de l'Espace Topologique
        if last_dist_topo is not None:
            corr_topo = np.corrcoef(dist_topo.flatten(), last_dist_topo.flatten())[0, 1]
            print(f" > Stabilité Espace Topologique (T vs T-1) : {corr_topo:.6f}")
        last_dist_topo = dist_topo.copy()
        
        # Nouvelle inférence de M0 brut via noyau gravitaire standard
        M0_raw = compute_nonDC_gravity_matrix(degrees, dist_topo)
        
        # --- PROJECTION ORTHOGONALE DE M0 PAR RAPPORT À M1 ---
        M1_centered = M1 - K
        M0_centered = M0_raw - K
        denom_project_0 = np.sum(M1_centered ** 2)
        if denom_project_0 > 1e-12:
            proj_factor_0 = np.sum(M1_centered * M0_centered) / denom_project_0
            M0_centered_ortho = M0_centered - (proj_factor_0 * M1_centered)
        else:
            M0_centered_ortho = M0_centered
            
        M0 = np.clip(K + M0_centered_ortho, 0.0, None)
        if np.sum(M0) > 0: M0 = M0 * (m2 / np.sum(M0))
        
        # =====================================================================
        # ÉVALUATION DE LA DÉCORRÉLATION ET DE L'INFORMATION MUTUELLE
        # =====================================================================
        current_rv = compute_matrix_rv_coefficient(M0, M1)
        current_mi = compute_mutual_information_spaces(X_spatial, X_topo)
        
        print(f" > Alignement des Modèles Nuls (RV M0 vs M1) : {current_rv:.4f}")
        print(f" > Information Mutuelle entre Espaces (MI)    : {current_mi:.6f}")
        
    print("\n=== Framework alterné continu terminé ===")
    return X_topo, X_spatial