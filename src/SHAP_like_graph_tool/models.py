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


def train_and_eval_xgboost(dataFrame, features=None, plot = False):
    all_stats = []

    if features == None : 
        X = dataFrame.drop('target', axis=1)
    else : 
        X = dataFrame[features]

    y = dataFrame['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        objective='binary:logistic',
        tree_method='hist', # Méthode plus stable sur beaucoup de systèmes
        n_jobs=1            # On force 1 seul thread pour éviter les conflits mémoire
    )
    
    model.fit(X_train, y_train)
            
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs > 0.5).astype(int)
    
    auc_roc = roc_auc_score(y_test, probs)
    ap = average_precision_score(y_test, probs)
    f1 = f1_score(y_test, preds)

    # AUC-PR (Aire sous la courbe Precision-Recall)
    precision, recall, _ = precision_recall_curve(y_test, probs)
    auc_pr = auc(recall, precision)

    if plot == True : 
        plot_confusion_matrix(y_test, preds)
        plot_probability_distribution(y_test, probs)

    # Compilation des résultats
    all_stats.append({
        'F1-Score': f1,
        'AUC-ROC': auc_roc,
        'AP': ap,
        'AUC-PR': auc_pr
    })
    
    print("\n RÉSULTATS")
    print(all_stats)

    all_stats = pd.DataFrame(all_stats)
    
    return all_stats, model, X_test, X_train, y_test, y_train

def evaluate_model_performance(model, X_test, y_test, plot=False):
    """
    Évalue un modèle déjà entraîné et retourne les métriques de performance.
    """
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs > 0.5).astype(int)
    
    # Calcul des métriques
    auc_roc = roc_auc_score(y_test, probs)
    ap = average_precision_score(y_test, probs)
    f1 = f1_score(y_test, preds)

    precision, recall, _ = precision_recall_curve(y_test, probs)
    auc_pr = auc(recall, precision)

    # Affichage des graphiques
    if plot:
        try:
            plot_confusion_matrix(y_test, preds)
            plot_probability_distribution(y_test, probs)
        except NameError:
            print("Fonctions de plot (confusion/distribution) non définies.")

    # Compilation
    stats = {
        'F1-Score': round(f1, 4),
        'AUC-ROC': round(auc_roc, 4),
        'AP': round(ap, 4),
        'AUC-PR': round(auc_pr, 4)
    }

    print("\n RÉSULTATS D'ÉVALUATION")
    print("-" * 30)
    for k, v in stats.items():
        print(f"{k:<10} : {v}")
    print("-" * 30)
    
    return pd.DataFrame([stats])

def plot_confusion_matrix(y_true, y_preds):
    # Calcul de la matrice
    cm = confusion_matrix(y_true, y_preds)
    
    # Création de l'affichage
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Pas de lien', 'Lien'])
    
    # On l'affiche avec un style soigné
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