import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


########################################
#### METHODE SiNE CUSTOM AC PAIRES #####
########################################

class PairwiseSignLoss(nn.Module):
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, z, pairs, signs):
        """
        z : Embeddings des noeuds (N, embedding_dim)
        pairs : Tenseur (M, 2) des paires (noeud_i, noeud_j)
        signs : Tenseur (M,) contenant le signe original (+1.0 ou -1.0)
        """
        if pairs.shape[0] == 0:
            return torch.tensor(0.0, requires_grad=True, device=z.device)
            
        anchors = z[pairs[:, 0]]
        neighbors = z[pairs[:, 1]]

        # Distance euclidienne entre les embeddings de la paire
        distances = torch.norm(anchors - neighbors, p=2, dim=1)

        # Si signe == 1  -> On minimise la distance
        loss_pos = distances
        
        # Si signe == -1 -> On veut distance >= margin
        loss_neg = torch.clamp(self.margin - distances, min=0.0)

        # Projection des signes [-1, 1] vers [0, 1] pour le masque
        is_pos = (signs + 1) / 2.0
        
        # Combinaison linéaire des pertes
        total_loss = is_pos * loss_pos + (1.0 - is_pos) * loss_neg

        return total_loss.mean()

def generate_pairwise_samples(R, num_samples_per_node=15, temperature=0.5):
    """
    R : Matrice d'adjacence ou de résidus (N, N)
    """
    pairs = []
    signs = []
    N = R.shape[0]
    
    for i in range(N):
        row = R[i]
        abs_residues = np.abs(row)
        
        # Masquage : le noeud i ne peut pas se piocher lui-même
        abs_residues_tensor = torch.tensor(abs_residues, dtype=torch.float32)
        abs_residues_tensor[i] = float('-inf')

        # Comptage du nb de voisins "valides" pour le noeud étudié (i)
        valid_candidates_mask = (abs_residues_tensor > 1e-6) & (abs_residues_tensor != float('-inf'))
        valid_indices = torch.where(valid_candidates_mask)[0].numpy()

        # GESTION DES NOEUDS ISOLÉS / COMPORTEMENT NORMAL
        if len(valid_indices) == 0:
            # Le noeud n'a aucun signal utile. On lui donne une probabilité uniforme 
            # sur TOUS les autres noeuds du graphe (pour éviter de bloquer l'algo)
            probs = np.ones(N) / (N - 1)
            probs[i] = 0.0
            # Le pool de choix devient tous les noeuds sauf i
            pool_to_sample = np.delete(np.arange(N), i)
        else:
            # Le noeud a du signal. On applique le Softmax UNIQUEMENT sur les candidats valides
            sub_scores = abs_residues_tensor[valid_indices]
            sub_probs = F.softmax(sub_scores / temperature, dim=0).numpy()
            
            # On reconstruit un vecteur de probabilité de taille N
            probs = np.zeros(N)
            probs[valid_indices] = sub_probs
            pool_to_sample = N

        probs_sum = probs.sum()
        if probs_sum > 0:
            normalization_ratio = 1/probs_sum
            probs = probs*normalization_ratio
            if abs(1-normalization_ratio)>1.05:
                print(f"ATTTENTION :SiNE probas normalisées par un facteur de {normalisation_ratio}")
            
        
       # TIRAGE : Le nombre maximum de tirages possibles sans remise est limité par notre pool de candidats réels
        available_candidates = len(valid_indices) if len(valid_indices) > 0 else (N - 1)
        eff_num_samples = min(num_samples_per_node, available_candidates)
        
        # TIRAGE ALÉATOIRE SANS REMISE
        sampled_nodes = np.random.choice(
            pool_to_sample, 
            size=eff_num_samples, 
            p=probs if isinstance(pool_to_sample, int) else None, 
            replace=False
        )
        
        # 6. ENREGISTREMENT DES PAIRES
        for j in sampled_nodes:
            pairs.append([i, j])
            # Signe réel : 1.0 pour les affinités/liens existants, -1.0 pour les inimitiés
            sign = 1.0 if R[i, j] >= 0 else -1.0
            signs.append(sign)
            
    return torch.tensor(pairs, dtype=torch.long), torch.tensor(signs, dtype=torch.float32)

def train_custom_signed_embedding(R_matrix, embedding_dim=64, epochs=100, lr=0.01, temperature=0.5):
    N = R_matrix.shape[0]
    
    # Initialisation des embeddings libres (Paramètre PyTorch)
    # Les noeuds isolés resteront proches de leur position d'initialisation 
    # car ils ne recevront aucun gradient significatif.
    node_embeddings = torch.nn.Parameter(torch.randn(N, embedding_dim) * 0.1)
    
    optimizer = torch.optim.Adam([node_embeddings], lr=lr)
    loss_fn = PairwiseSignLoss(margin=2.0)
    
    for epoch in range(epochs):
        # 1. Échantillonnage stochastique custom
        pairs, signs = generate_pairwise_samples(R_matrix, num_samples_per_node=15, temperature=temperature)
        
        optimizer.zero_grad()
        
        # 2. Calcul de la perte sur les paires échantillonnées
        loss = loss_fn(node_embeddings, pairs, signs)
        
        # 3. Rétropropagation
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")
            
    return node_embeddings.detach().numpy()

##############################################
##### METHODE SiNE ORIGINALE AC TRIPLETS #####
##############################################


class OriginalSiNETripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, z, triplets):
        """
        z : Embeddings des noeuds (N, embedding_dim)
        triplets : Tenseur (M, 3) des triplets (ancre_i, ami_j, ennemi_k)
        """
        if triplets.shape[0] == 0:
            return torch.tensor(0.0, requires_grad=True, device=z.device)
            
        anchors = z[triplets[:, 0]]
        friends = z[triplets[:, 1]]
        enemies = z[triplets[:, 2]]

        # Calcul des distances euclidiennes (norme L2) conforme à SiNE
        dist_to_friend = torch.norm(anchors - friends, p=2, dim=1)
        dist_to_enemy = torch.norm(anchors - enemies, p=2, dim=1)

        # Objective function native : max(0, dist(i,j) - dist(i,k) + margin)
        loss = torch.clamp(dist_to_friend - dist_to_enemy + self.margin, min=0.0)

        return loss.mean()

def generate_sine_triplets(R, num_triplets_per_node=15, temperature=0.5):
    """
    Génère les triplets (Ancre, Voisin_Positif, Voisin_Négatif) originaux du modèle SiNE.
    """
    triplets = []
    N = R.shape[0]
    
    for i in range(N):
        row = R[i]
        
        # 1. Extraction séparée des candidats Positifs et Négatifs
        pos_mask = (row >= 0)
        neg_mask = (row < 0)
        
        # On interdit le self-loop
        pos_mask[i] = False
        neg_mask[i] = False
        
        pos_indices = np.where(pos_mask)[0]
        neg_indices = np.where(neg_mask)[0]
        
        # 2. GESTION DES NOEUDS ISOLÉS / ASYMÉTRIQUES
        # SiNE requiert impérativement au moins un positif ET un négatif pour créer un triplet.
        # Si l'une des deux populations manque, on bascule sur un repli probabiliste uniforme.
        if len(pos_indices) == 0 or len(neg_indices) == 0:
            # On génère un pool de repli uniforme (tous les noeuds sauf i)
            uniform_pool = np.delete(np.arange(N), i)
            eff_samples = min(num_triplets_per_node, len(uniform_pool) // 2)
            
            if eff_samples == 0:
                continue
                
            # On sépare arbitrairement le tirage uniforme pour simuler des partenaires
            sampled_nodes = np.random.choice(uniform_pool, size=eff_samples * 2, replace=False)
            for idx in range(eff_samples):
                triplets.append([i, sampled_nodes[idx], sampled_nodes[idx + eff_samples]])
            continue

        # 3. ÉCHANTILLONNAGE PAR SOFTMAX SÉPARÉ (Respect des intensités)
        # Échantillonnage des Amis (Plus le résidu est grand/positif, plus on le pioche)
        pos_scores = torch.tensor(row[pos_indices], dtype=torch.float32)
        pos_probs = F.softmax(pos_scores / temperature, dim=0).numpy().astype(np.float64)
        pos_probs /= pos_probs.sum()  # Sécurité flottants
        
        # Échantillonnage des Ennemis (Plus le résidu est négatif, donc plus sa valeur absolue est grande, plus on le pioche)
        neg_scores = torch.tensor(-row[neg_indices], dtype=torch.float32)  # -row pour inverser le signe négatif
        neg_probs = F.softmax(neg_scores / temperature, dim=0).numpy().astype(np.float64)
        neg_probs /= neg_probs.sum()  # Sécurité flottants

        # 4. TIRAGE SANS REMISE POUNDÉRÉ
        eff_pos_samples = min(num_triplets_per_node, len(pos_indices))
        eff_neg_samples = min(num_triplets_per_node, len(neg_indices))
        eff_triplets = min(eff_pos_samples, eff_neg_samples)
        
        sampled_friends = np.random.choice(pos_indices, size=eff_triplets, p=pos_probs, replace=False)
        sampled_enemies = np.random.choice(neg_indices, size=eff_triplets, p=neg_probs, replace=False)
        
        # 5. ASSEMBLEMENT DES TRIPLETS SINE
        for f, e in zip(sampled_friends, sampled_enemies):
            triplets.append([i, f, e])
            
    return torch.tensor(triplets, dtype=torch.long)

def train_original_sine_embedding(R_matrix, embedding_dim=64, epochs=100, lr=0.01, temperature=0.5):
    N = R_matrix.shape[0]
    
    # Initialisation uniforme des embeddings libres
    node_embeddings = torch.nn.Parameter(torch.randn(N, embedding_dim) * 0.1)
    
    optimizer = torch.optim.Adam([node_embeddings], lr=lr)
    loss_fn = OriginalSiNETripletLoss(margin=1.0)  # La marge d'origine de SiNE est souvent fixée à 1.0
    
    for epoch in range(epochs):
        # 1. Échantillonnage stochastique par Triplet
        triplets = generate_sine_triplets(R_matrix, num_triplets_per_node=15, temperature=temperature)
        
        optimizer.zero_grad()
        
        # 2. Calcul du coût de structure triplet
        loss = loss_fn(node_embeddings, triplets)
        
        # 3. Optimisation
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Original SiNE Loss: {loss.item():.4f}")
            
    return node_embeddings.detach().numpy()