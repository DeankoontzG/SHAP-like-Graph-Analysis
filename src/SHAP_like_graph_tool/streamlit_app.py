import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import shap
from pathlib import Path

st.set_page_config(page_title="GraphLink Expert - Live SHAP", layout="wide")

# --- 1. FONCTIONS DE CALCUL (DÉFINIES AU DÉBUT) ---

def compute_live_shap(model, rows, masker_data):
    """Calcule le SHAP pour une ou plusieurs lignes (format matrice)."""
    f = lambda x: model.predict_proba(x)[:, 1]
    explainer = shap.Explainer(f, masker_data)
    shap_result = explainer(rows)
    vals = shap_result.values
    # Gestion dimension classe binaire
    if len(vals.shape) == 3: vals = vals[:, :, 1]
    return vals

@st.cache_data
def get_avg_top_k_shap(graph_name, k_value, _model, _top_k_df, _masker_sample):
    cols_to_exclude = ['prob', 'target']
    clean_sample = _top_k_df.drop(columns=[c for c in cols_to_exclude if c in _top_k_df.columns])
    
    n_samples = min(50, len(clean_sample))
    sample_to_explain = clean_sample.sample(n=n_samples, random_state=42)
    
    vals = compute_live_shap(_model, sample_to_explain, _masker_sample)
    return vals.mean(axis=0)

@st.cache_resource
def load_essentials(g_name):
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    data_path = project_root / "outputs" / "results" / f"xgboost_data_{g_name}.joblib"
    if not data_path.exists():
        st.error(f"Fichier introuvable : {data_path}")
        return None
    return joblib.load(data_path)

# --- 2. CONFIGURATION ET INITIALISATION ---

feature_groups = {
    "Topologie": ['cn', 'aa', 'jc', 'pa', 'sp'],
    "Centralité": ['pr_u', 'pr_v', 'lcc_u', 'lcc_v', 'dc_u', 'dc_v', 'and_u', 'and_v'],
    "Communauté": ['louvain_density', 'same_louvain', 'infomap_density', 'same_infomap', 'sbm_density', 'same_sbm'],
    "Embeddings": ['n2v_homophily_cos', 'n2v_homophily_dist', 'deepwalk_cos', 'deepwalk_dist']
}

# --- GÉNÉRATION DES OPTIONS DE GRAPHES ---
def get_graph_names():
    names = []
    for i in [0.0, 0.25, 0.5, 0.75, 1.0]:
        pos = 1.0 - i
        s_i = f"{i:.2f}".replace('.', '_')
        s_p = f"{pos:.2f}".replace('.', '_')
        names.append(f"artificial_graph_sbmv2_{s_i}_pos_{s_p}")
        
    for i_int in range(0, 105, 5):
        i = i_int / 100.0
        pos = 1.0 - i
        s_i = f"{i:.2f}".replace('.', '_')
        s_p = f"{pos:.2f}".replace('.', '_')
        names.append(f"artificial_graph_sbm_{s_i}_pos_{s_p}")
    
    return names

graph_options = get_graph_names()

# --- SIDEBAR : SÉLECTION DU PROJET ---
st.sidebar.header("📁 Configuration du Graphe")
selected_graph = st.sidebar.selectbox(
    "Choisir le graphe à analyser :",
    options=graph_options,
    index=graph_options.index("artificial_graph_sbmv2_0_00_pos_1_00") # Valeur par défaut
)
data = load_essentials(selected_graph)

if data:
    # Définition des variables de base
    X_hidden = data['X_hidden']
    y_hidden = data['y_hidden']
    model = data['model']

    # Calcul des scores globaux
    probs = model.predict_proba(X_hidden)[:, 1]
    results_df = X_hidden.copy()
    results_df['prob'] = probs
    results_df['target'] = y_hidden.values
    results_df = results_df.sort_values(by='prob', ascending=False)

    # Sidebar pour le choix du K
    k = st.sidebar.slider("Nombre de prédictions (Top-K)", 10, 500, 100)
    top_k_df = results_df.head(k)

    # --- 3. PRÉPARATION DES RÉFÉRENCES (Moyenne du Top-K) ---
    # On crée un masker fixe pour tout le monde
    masker_sample = X_hidden.sample(n=min(100, len(X_hidden)), random_state=42)
    
    with st.spinner("Calcul de la signature moyenne du Top-K..."):
        avg_shap_values = get_avg_top_k_shap(selected_graph, k, model, top_k_df, masker_sample)

    # --- 4. INTERFACE PRINCIPALE ---
    st.title(f"🔍 Analyse Live : {selected_graph}")
    
    col_list, col_shap = st.columns([1, 1.5])

    with col_list:
        st.subheader(f"Top {k} Opportunités")
        selected_idx = st.selectbox(
            "Sélectionner une paire :",
            options=top_k_df.index,
            format_func=lambda i: f"Rang {top_k_df.index.get_loc(i)+1} - Score: {top_k_df.loc[i, 'prob']:.3f}"
        )
        st.write("**Valeurs brutes des features :**")
        st.dataframe(X_hidden.loc[[selected_idx]].T)


    with col_shap:
        row_to_explain = X_hidden.loc[[selected_idx]]
        if 'prob' in row_to_explain.columns:
            row_to_explain = row_to_explain.drop(columns=['prob', 'target'], errors='ignore')
        pair_shap_values = compute_live_shap(model, row_to_explain, masker_sample)[0]
        feature_names = X_hidden.columns.tolist()

        # --- RADAR 1 : DÉTAILLÉ (NORMALISÉ) ---
        st.subheader("Signature Détaillée (Normalisée via √)")
        def norm(arr): return np.sign(arr) * np.sqrt(np.abs(arr))
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatterpolar(r=norm(pair_shap_values), theta=feature_names, fill='toself', name='Cette Paire'))
        fig1.add_trace(go.Scatterpolar(r=norm(avg_shap_values), theta=feature_names, name='Moyenne Top-K', line=dict(dash='dot', color='gray')))
        fig1.update_layout(polar=dict(radialaxis=dict(visible=True, showticklabels=False)), height=500)
        st.plotly_chart(fig1, use_container_width=True)

        # --- RADAR 2 : PAR GROUPES (SOMME BRUTE) ---
        st.subheader("Signature par Familles (Impact Total)")
        group_names = list(feature_groups.keys())
        group_vals_pair = []
        group_vals_avg = []

        for g_name, cols in feature_groups.items():
            idxs = [feature_names.index(c) for c in cols if c in feature_names]
            group_vals_pair.append(pair_shap_values[idxs].sum())
            group_vals_avg.append(avg_shap_values[idxs].sum())

        fig2 = go.Figure()
        fig2.add_trace(go.Scatterpolar(r=group_vals_pair, theta=group_names, fill='toself', name='Cette Paire', fillcolor='rgba(255, 75, 75, 0.4)', line=dict(color='#FF4B4B')))
        fig2.add_trace(go.Scatterpolar(r=group_vals_avg, theta=group_names, name='Moyenne Top-K', line=dict(dash='dot', color='gray')))
        fig2.update_layout(polar=dict(radialaxis=dict(visible=True)), height=400)
        st.plotly_chart(fig2, use_container_width=True)

        # --- METRICS ---
        cols_stats = st.columns(len(group_names))
        for i, g in enumerate(group_names):
            diff = group_vals_pair[i] - group_vals_avg[i]
            cols_stats[i].metric(g, f"{group_vals_pair[i]:.2f}", f"{diff:+.2f}")