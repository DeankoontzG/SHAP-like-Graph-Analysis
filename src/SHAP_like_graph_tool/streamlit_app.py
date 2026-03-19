import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import shap
from pathlib import Path

st.set_page_config(page_title="GraphLink Expert - Live SHAP", layout="wide")

# --- 1. FONCTIONS DE CALCUL (DÉFINIES AU DÉBUT) ---

def compute_live_shap(model, masker_data):
    """Calcule le SHAP pour une ou plusieurs lignes (format matrice)."""
    explainer = shap.TreeExplainer(
        model, 
        data=masker_data, 
        model_output="raw",
        feature_perturbation="interventional"
    )
    
    return explainer

def get_interaction_heatmap(inter_matrix, feature_names, top_indices, title):
    sub_matrix = inter_matrix[top_indices][:, top_indices]
    mask = np.triu(np.ones_like(sub_matrix, dtype=bool))
    sub_matrix_masked = np.where(mask, sub_matrix, np.nan)
    sub_features = [feature_names[i] for i in top_indices]

    fig = go.Figure(data=go.Heatmap(
        z=sub_matrix_masked,
        x=sub_features,
        y=sub_features,
        colorscale='RdBu_r',
        zmid=0,
        texttemplate="%{z:.3f}",
        textfont={"size": 10},
        colorbar=dict(thickness=15, len=0.5, yanchor="middle") 
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=20)),
        height=650,
        margin=dict(l=150, r=50, t=100, b=150), 
        xaxis=dict(
            side='bottom', 
            tickangle=-45, # Angle négatif pour mieux lire de gauche à droite
            fixedrange=True,
            constrain="domain" 
        ),
        yaxis=dict(
            autorange='reversed', 
            fixedrange=True,
            scaleanchor=None 
        ),
        autosize=True
    )
    return fig

@st.cache_data
def get_avg_dataset_shap(graph_name, k_value, _explainer, dataset):
    cols_to_exclude = ['prob', 'target']
    clean_sample = dataset.drop(columns=[c for c in cols_to_exclude if c in dataset.columns])
    
    shap_result = _explainer(clean_sample)
    vals = shap_result.values
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
    "Topologie": ['cn', 'aa', 'jc', 'pa', 'sp', 'ra'],
    "Centralité": ['pr_u', 'pr_v','ppr_u', 'ppr_v', 'lcc_u', 'lcc_v', 'dc_u', 'dc_v', 'and_u', 'and_v',
                   'katz_u', 'katz_v'],
    "Communauté": ['louvain_density', 'same_louvain', 'infomap_density', 'same_infomap', 'sbm_density', 'same_sbm',
                   'leiden_density', 'same_leiden', "surprise_density", "same_surprise", "significance_density", "same_significance"],
    "Embeddings": ["deepwalk_cos", "deepwalk_rank", "deepwalk_had_mean",  "deepwalk_had_std", "deepwalk_dist", "deepwalk_dist_sq",
                   "n2v_homophily_cos", "n2v_homophily_rank", "n2v_homophily_had_mean",  "n2v_homophily_had_std", "n2v_homophily_dist", "n2v_homophily_dist_sq"]
}

# --- GÉNÉRATION DES OPTIONS DE GRAPHES ---
def get_graph_names():

    names = ["reel_blumenau_drug",
             "reel_facebook_friends",
             "reel_cintestinalis",
             "reel_faculty_hiring_computer_science", 
             "reel_jazz_collab",
             "reel_wiki_science",
             "reel_Airports"]

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
    shap_explainer = compute_live_shap(model, masker_data=masker_sample)
    
    with st.spinner("Calcul de la signature moyenne du Top-K..."):
        avg_shap_values_top_k = get_avg_dataset_shap(selected_graph, k, shap_explainer, top_k_df)
    with st.spinner("Calcul de la signature moyenne du dataset global .."):    
        avg_shap_values_global = get_avg_dataset_shap(selected_graph, k, shap_explainer, results_df.sample(n=1000))

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
        pair_shap_values = get_avg_dataset_shap(selected_graph, k, shap_explainer, row_to_explain)
        feature_names = X_hidden.columns.tolist()

        all_features = X_hidden.columns.tolist()
        grouped_features = [item for sublist in feature_groups.values() for item in sublist]
        missing = set(all_features) - set(grouped_features)

        if missing:
            st.warning(f"⚠️ Colonnes oubliées dans les groupes : {missing}")

        # --- RADAR 1 : DÉTAILLÉ (NORMALISÉ) ---
        st.subheader("Signature Détaillée (Impact Réel)")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatterpolar(r=pair_shap_values, theta=feature_names, fill='toself', name='Cette Paire'))
        fig1.add_trace(go.Scatterpolar(r=avg_shap_values_top_k, theta=feature_names, name='Moyenne Top-K', line=dict(dash='dot', color='gray')))
        fig1.add_trace(go.Scatterpolar(r=avg_shap_values_global, theta=feature_names, name='Moyenne dataset', line=dict(dash='dot', color='blue')))
        fig1.update_layout(polar=dict(radialaxis=dict(visible=True, showticklabels=True, tickfont=dict(size=10, color='gray'), tickangle=45, gridcolor="lightgray")), height=500, showlegend=True)        
        st.plotly_chart(fig1, use_container_width=True)

        # --- RADAR 2 : PAR GROUPES (SOMME BRUTE) ---
        st.subheader("Signature par Familles (Impact Total)")
        group_names = list(feature_groups.keys())
        group_vals_pair = []
        group_vals_avg_top_k = []
        group_vals_avg_global = []

        for g_name, cols in feature_groups.items():
            idxs = [feature_names.index(c) for c in cols if c in feature_names]
            group_vals_pair.append(pair_shap_values[idxs].sum())
            group_vals_avg_top_k.append(avg_shap_values_top_k[idxs].sum())
            group_vals_avg_global.append(avg_shap_values_global[idxs].sum())

        fig2 = go.Figure()
        fig2.add_trace(go.Scatterpolar(r=group_vals_pair, theta=group_names, fill='toself', name='Cette Paire'))
        #fig2.add_trace(go.Scatterpolar(r=group_vals_pair, theta=group_names, fill='toself', name='Cette Paire', fillcolor='rgba(255, 75, 75, 0.4)', line=dict(color='#FF4B4B')))
        fig2.add_trace(go.Scatterpolar(r=group_vals_avg_top_k, theta=group_names, name='Moyenne Top-K', line=dict(dash='dot', color='gray')))
        fig2.add_trace(go.Scatterpolar(r=group_vals_avg_global, theta=group_names, name='Moyenne dataset', line=dict(dash='dot', color='blue')))
        fig2.update_layout(polar=dict(radialaxis=dict(visible=True, showticklabels=True, tickfont=dict(size=10, color='gray'), tickangle=45, gridcolor="lightgray")), height=400, showlegend=True)        

        st.plotly_chart(fig2, use_container_width=True)

        # --- METRICS ---
        cols_stats = st.columns(len(group_names))
        base_val = shap_explainer(row_to_explain).base_values[0]

        for i, g in enumerate(group_names):
            diff_top_k = group_vals_pair[i] - group_vals_avg_top_k[i]
            diff_global = group_vals_pair[i] - group_vals_avg_global[i]
            
            with cols_stats[i]:
                st.metric(label=f"Impact {g}", 
                    value=f"{group_vals_pair[i]:.2f}", 
                    delta=f"{diff_top_k:+.2f} (vs Top-K)",
                    delta_color="normal" )

                st.metric(label="", value="", delta=f"{diff_global:+.2f} (vs Global)")   

        with cols_stats[0]:
            st.metric(
                label="Probabilité de base", 
                value=f"{base_val:.2f}",
                help="Il s'agit de la probabilité moyenne du dataset de référence (le point de départ du modèle)."
            )
            st.caption("Score de départ") 


        # --- ANALYSE DES INTERRACTIONS ENTRE FEATURES ---

        st.divider()
        st.subheader("Analyse Comparative des Synergies entre features")

        # On recréé un explainer compatible ac calcul des shap_interaction_values
        booster = model.get_booster()
        tree_explainer = shap.TreeExplainer(booster)

        cols_to_drop = ['prob', 'target']
        X_pair = row_to_explain.drop(columns=cols_to_drop, errors='ignore')
        X_top_k = top_k_df.drop(columns=cols_to_drop, errors='ignore')
        X_global = results_df.drop(columns=cols_to_drop, errors='ignore').sample(n=min(50, len(results_df)))

        inter_pair_raw = tree_explainer.shap_interaction_values(X_pair)[0]
        inter_avg_top_k_raw = tree_explainer.shap_interaction_values(X_top_k).mean(axis=0)
        inter_avg_global_raw = tree_explainer.shap_interaction_values(X_global).mean(axis=0)
       
        top_n = 10
        importance = np.abs(inter_pair_raw).sum(axis=1)
        top_idx = np.argsort(importance)[-top_n:]

        # 4. Affichage avec ta fonction get_interaction_heatmap
        tab1, tab2, tab3 = st.tabs(["🎯 Cette Paire", "📊 Moyenne Top-K", "🌐 Moyenne Global"])

        with tab1:
            st.plotly_chart(get_interaction_heatmap(inter_pair_raw, feature_names, top_idx, "Synergies Locales"), use_container_width=True)
        with tab2:
            st.plotly_chart(get_interaction_heatmap(inter_avg_top_k_raw, feature_names, top_idx, "Synergies Top-K"), use_container_width=True)
        with tab3:
            st.plotly_chart(get_interaction_heatmap(inter_avg_global_raw, feature_names, top_idx, "Synergies Globales"), use_container_width=True)