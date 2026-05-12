import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score


########################################
######## DEFINITION DES MODELES ########
########################################

class GAE(nn.Module):
    """Encodeur de graphe classique (GCN)."""
    def __init__(self, in_channels, out_channels):
        super(GAE, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    @torch.no_grad()
    def compute_link_loss(self, z, pos_edge_index, neg_edge_index):
        """Calcule la BCE locale via produit scalaire pour diagnostic."""
        pos_src, pos_dst = pos_edge_index
        neg_src, neg_dst = neg_edge_index
        pos_logits = (z[pos_src] * z[pos_dst]).sum(dim=-1)
        neg_logits = (z[neg_src] * z[neg_dst]).sum(dim=-1)
        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
        return F.binary_cross_entropy_with_logits(logits, labels).item()


class GeoEncoder(nn.Module):
    """Encodeur géométrique utilisant des projections de Fourier."""
    def __init__(self, in_pos_dim, out_channels, freq_dim=16, scale=1):
        super(GeoEncoder, self).__init__()
        self.in_channels = in_pos_dim
        self.out_channels = out_channels
        # On projette les positions (ex: 2D) vers un espace plus large via Fourier
        self.freq_dim = freq_dim
        self.scale = scale
        self.register_buffer('frequencies', torch.randn(in_pos_dim, self.freq_dim) * self.scale)
        
        self.mlp = nn.Sequential(
            nn.Linear(self.freq_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, out_channels)
        )

    def forward(self, pos):
        # encodage de Fourier : [sin(f*x), cos(f*x)]
        proj = pos @ self.frequencies
        x = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.mlp(x)

    @torch.no_grad()
    def compute_link_loss(self, z, pos_edge_index, neg_edge_index):
        """Calcule la BCE locale via produit scalaire pour diagnostic."""
        pos_src, pos_dst = pos_edge_index
        neg_src, neg_dst = neg_edge_index
        pos_logits = (z[pos_src] * z[pos_dst]).sum(dim=-1)
        neg_logits = (z[neg_src] * z[neg_dst]).sum(dim=-1)
        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
        return F.binary_cross_entropy_with_logits(logits, labels).item()


class MLPLinkPredictor(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, z_u, z_v):
        edge_feat = z_u * z_v # Porduit de hammard pour l'arête.
        return self.mlp(edge_feat)


class JointLinkPredictionModel(nn.Module):
    def __init__(self, encoder_a, encoder_b):
        super().__init__()
        self.encoder_a = encoder_a
        self.encoder_b = encoder_b
        
        self.decoder = MLPLinkPredictor(self.encoder_a.out_channels + self.encoder_b.out_channels)

    def encode(self, data):
        # Gestion flexible des entrées selon le type d'encodeur
        # On regarde si l'encodeur a besoin de (x, edge_index) ou juste de (pos)
        if isinstance(self.encoder_a, GAE):
            z_a = self.encoder_a(data.x, data.edge_index)
        elif isinstance(self.encoder_a, GeoEncoder): # GeoEncoder
            z_a = self.encoder_a(data.pos)
        else :
            raise ValueError(f"Erreur : encoder a n'est pas un GAE ni un GeoEncoder")
            
        if isinstance(self.encoder_b, GAE):
            z_b = self.encoder_b(data.x, data.edge_index)
        elif isinstance(self.encoder_b, GeoEncoder): # GeoEncoder
            z_b = self.encoder_b(data.pos)
        else :
            raise ValueError(f"Erreur : encoder b n'est pas un GAE ni un GeoEncoder")
            
        return z_a, z_b

    def forward(self, data, edge_index):
        z_a, z_b = self.encode(data)
        
        # On récupère les embeddings des sources et destinations
        src, dst = edge_index
        
        # Concaténation des représentations d'arêtes
        # [z_a_u * z_a_v || z_b_u * z_b_v]
        z_combined_u = torch.cat([z_a[src], z_b[src]], dim=-1)
        z_combined_v = torch.cat([z_a[dst], z_b[dst]], dim=-1)
        
        return self.decoder(z_combined_u, z_combined_v)
    
    def fit(self, data,  epochs=200, neg_sample_ratio=10, lr=0.01, lambda_ortho=1.0, disentangle=True):
        """Entraîne simultanément encoder_a, encoder_b et le decoder MLP."""
        # On passe self.parameters() pour inclure A, B et le MLP du décodeur
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            neg_edge_index = negative_sampling(
                data.edge_index, 
                num_nodes= data.num_nodes,
                num_neg_samples= data.edge_index.size(1) * neg_sample_ratio 
            )

            pos_out = self.forward(data, data.edge_index)
            neg_out = self.forward(data, neg_edge_index)
            
            # Calcul de la reconstruction (BCE)
            logits = torch.cat([pos_out, neg_out]).squeeze()
            labels = torch.cat([
                torch.ones(pos_out.size(0)), 
                torch.zeros(neg_out.size(0))
            ]).to(logits.device)
            
            loss_rec = F.binary_cross_entropy_with_logits(logits, labels)
            
            # Calcul de l'orthogonalité (si activé)
            loss_ortho = torch.tensor(0.0).to(logits.device)
            if disentangle:
                z_a, z_b = self.encode(data)
                z_a_n = F.normalize(z_a, p=2, dim=1)
                z_b_n = F.normalize(z_b, p=2, dim=1)
                loss_ortho = torch.norm(torch.mm(z_a_n.t(), z_b_n), p='fro')
            
            # Backprop globale
            total_loss = loss_rec + (lambda_ortho * loss_ortho)
            total_loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                z_a, z_b = self.encode(data)
                loss_a = self.encoder_a.compute_link_loss(z_a, data.edge_index, neg_edge_index)
                loss_b = self.encoder_b.compute_link_loss(z_b, data.edge_index, neg_edge_index)
                mode = "DISENTANGLED" if disentangle else "CLASSIC"
                print(f"[{mode}] Ep {epoch:03d} | Loss: {total_loss.item():.4f} | Rec: {loss_rec.item():.4f} | Lambda*Ortho: {lambda_ortho*loss_ortho.item():.4f}")
                print(f"Détails modèles | A: {loss_a:.3f} | B: {loss_b:.3f}")


    @torch.no_grad()
    def evaluate(self, test_data, neg_sample_ratio=10):
        """Évalue le modèle sur un set de test/validation."""

        self.eval()
        
        pos_out = self.forward(test_data, test_data.edge_index)
        pos_probs = torch.sigmoid(pos_out).squeeze()
        
        neg_edge_index = negative_sampling(
            edge_index=test_data.edge_index,
            num_nodes=test_data.num_nodes,
            num_neg_samples=int(test_data.edge_index.size(1) * neg_sample_ratio)
        )
        neg_out = self.forward(test_data, neg_edge_index)
        neg_probs = torch.sigmoid(neg_out).squeeze()
        
        y_true = torch.cat([
            torch.ones(pos_probs.size(0)),
            torch.zeros(neg_probs.size(0))
        ]).cpu().numpy()
    
        y_pred = torch.cat([pos_probs, neg_probs]).cpu().numpy()
        
        auc = roc_auc_score(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)
        
        return auc, ap
