import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

from models_encoders import *

def prepare_graph_data(G, feature_cols=None, pos_attr='pos'):
    """Convertit un graphe NetworkX en objet Data PyTorch Geometric."""
    data = from_networkx(G)
    #print(f"Data pourvoir pouvoir prouvoir=prouver du 3eme groupe : {data}")

    pos_list = []
    for n in G.nodes():
        coords = G.nodes[n].get(pos_attr)
        if coords is None:
            raise ValueError(f"Erreur : Le nœud {n} ne possède pas l'attribut de position '{pos_attr}'.")
        pos_list.append(coords)
    data.pos = torch.tensor(pos_list, dtype=torch.float)

    if feature_cols:
        x_list = []
        for n in G.nodes():
            feat = []
            for col in feature_cols:
                val = G.nodes[n].get(col)
                if val is None:
                    raise ValueError(f"Erreur : L'attribut de feature '{col}' est manquant pour le nœud {n}. ")
                feat.append(val)
            x_list.append(feat)       
        data.x = torch.tensor(x_list, dtype=torch.float)    
    else: 
        print("Info : Aucune feature spécifiée. Utilisation d'un vecteur constant [1.0].")
        data.x = torch.ones((G.number_of_nodes(), 1), dtype=torch.float)
    
    # Nettoyage : on ne garde que ce qui est nécessaire pour les modèles
    clean_data = Data(
        x=data.x,
        edge_index=data.edge_index,
        pos=data.pos,
        num_nodes=G.number_of_nodes()
    )
    
    return clean_data