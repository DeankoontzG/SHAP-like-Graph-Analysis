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

# nohup python -u main.py 2>&1 | grep --line-buffered -vE "it/s|%|\[.*\]" | grep --line-buffered "." > myoutfile.log &

def load_graphml_safe(path):
        with open(path, 'r', encoding='utf-8') as f:
            raw_data = f.read()

        clean_data = html.unescape(raw_data)
        G = nx.read_graphml(io.StringIO(clean_data))
        
        print(f"✅ Graphe chargé : {G.number_of_nodes()} nœuds et {G.number_of_edges()} liens.")
        return G

if __name__ == "__main__":

    names_list = [#"blumenau_drug",
                  "facebook_friends",
                  "cintestinalis",
                  "faculty_hiring_computer_science", 
                  "jazz_collab",
                  "wiki_science"
                  #"Airports"
    ]
                  
    
    for name in names_list:
        print("######################################")
        print(f"#### graph {name} :  ####")
        print("######################################")
        G_name = f"reel_{name}"  
        path = f"graph_library/{G_name}.graphml"
        print(G_name)
    
        try:
            G = load_graphml_safe(path)
            print(f"Graphe chargé avec succès : {G.number_of_nodes()} nœuds et {G.number_of_edges()} liens.")
        except Exception as e:
            print(f"Erreur lors de la conversion : {e}")
    
        gp.execute(G, G_name)

    for name in names_list:
        print("######################################")
        print(f"#### graph {name} :  ####")
        print("######################################")
        G_name = f"reel_{name}"  
        path = f"graph_library/{G_name}.graphml"
        print(G_name)
    
        try:
            G = load_graphml_safe(path)
            print(f"Graphe chargé avec succès : {G.number_of_nodes()} nœuds et {G.number_of_edges()} liens.")
        except Exception as e:
            print(f"Erreur lors de la conversion : {e}")
    
        gp.evaluate(G_name)