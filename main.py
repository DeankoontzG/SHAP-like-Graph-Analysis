import src.SHAP_like_graph_tool as gp
import src.SHAP_like_graph_tool.utils as gput

import networkx as nx
import html
import io
import json
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
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

if __name__ == "__main__":
    
    for i in np.arange(0.00, 1.25, 0.25):
        print("######################################")
        print(f"#### graph sbm v2 {i:.2f} pos {1-i:.2f} ####")
        print("######################################")
        G_name = f"artificial_graph_sbmv2_{f'{i:.2f}'.replace('.', '_')}_pos_{f'{1-i:.2f}'.replace('.', '_')}"
        print(G_name)
    
        with open(f"graph_library/{G_name}.json", 'r', encoding='utf-8') as f:
            data = json.load(f)  
    
        try:
            G = nx.node_link_graph(data)
            print(f"Graphe chargé avec succès : {G.number_of_nodes()} nœuds et {G.number_of_edges()} liens.")
        except Exception as e:
            print(f"Erreur lors de la conversion : {e}")
    
        gp.evaluate(G_name)