from xgboost import XGBClassifier
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                             precision_recall_curve, auc, f1_score, 
                             confusion_matrix, ConfusionMatrixDisplay)
#import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os


def train_and_test_xgboost(dataFrame, features=None, plot=False, max_depth=6, learning_rate=0.1, min_child_weight=10, nb_estimators=100):
    X = dataFrame[features] if features else dataFrame.drop(["target", "u", "v", "label"], axis=1)
    y = dataFrame['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
            
    model = XGBClassifier(
        nb_estimators=nb_estimators, learning_rate=learning_rate, max_depth=max_depth,
        min_child_weight=min_child_weight, objective='binary:logistic', tree_method='hist', n_jobs=1
    )
    model.fit(X_train, y_train)
    
    test_stats = get_performance_metrics(model, X_test, y_test, prefix="Test_")

    if plot == True : 
        probs = model.predict_proba(X)[:, 1]
        preds = (probs > 0.5).astype(int)
        plot_confusion_matrix(y_test, preds)
        plot_probability_distribution(y_test, probs)
    
    return test_stats, model, X_train, y_train, X_test, y_test


def get_performance_metrics(model, X, y, prefix=""):
    probs = model.predict_proba(X)[:, 1]
    preds = (probs > 0.5).astype(int)
    
    precision, recall, _ = precision_recall_curve(y, probs)

    stats = {
        f'{prefix}AP': average_precision_score(y, probs),
        f'{prefix}AUC-ROC': roc_auc_score(y, probs),
        f'{prefix}F1-Score': f1_score(y, preds),
        f'{prefix}AUC-PR': auc(recall, precision)
    }
    
    return pd.DataFrame([stats])

def plot_confusion_matrix(y_true, y_preds):
    cm = confusion_matrix(y_true, y_preds)
    
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Pas de lien', 'Lien'])
    
    disp.plot(cmap='Blues', values_format='d') # 'd' pour les nombres entiers
    
    plt.title(f"Matrice de Confusion")
    plt.grid(False) # On enlève la grille qui gêne sur une heatmap
    plt.savefig("outputs/plots/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_probability_distribution(y_true, y_probs):

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    plt.figure(figsize=(10, 6))

    if len(y_probs[y_true == 1]) > 0:
        sns.histplot(y_probs[y_true == 1], color="green", label="Vrais Liens (Positifs)", 
                     kde=True, stat="density", alpha=0.5)
    
    if len(y_probs[y_true == 0]) > 0:
        sns.histplot(y_probs[y_true == 0], color="red", label="Faux Liens (Négatifs)", 
                     kde=True, stat="density", alpha=0.5)
    
    plt.axvline(x=0.5, color='black', linestyle='--', label='Seuil F1 (0.5)')
    plt.title(f"Distribution des Probabilités")
    plt.xlabel("Probabilité prédite par XGBoost")
    plt.ylabel("Densité")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("outputs/plots/distribution_probabilites.png", dpi=300, bbox_inches='tight')
    plt.show()