"""
Graph Intelligence Dashboard — OGBN-Arxiv Node Classification
Task 08 deliverable.

Displays graph statistics, model performance, live node classification
predictions, and embedding visualizations for the three trained GNN
models (GCN, GraphSAGE, GAT) built in this project.

Run with:
    streamlit run dashboard/app.py
"""

import sys
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import torch


# ============================================================
# PAGE CONFIG — must be the first Streamlit call
# ============================================================

st.set_page_config(
    page_title="Graph Intelligence — OGBN-Arxiv",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DESIGN TOKENS
# ============================================================
# A small, deliberate palette — white/black base with five named
# accents, each with a fixed meaning used consistently throughout:
#   purple  -> brand / primary / navigation
#   green   -> good performance, high confidence, positive signal
#   orange  -> moderate / caution / secondary metric
#   red     -> weak performance, low confidence, needs attention
#   rose    -> highlights, bonus features, callouts

INK = "#1D1D1F"
INK_SOFT = "#6E6E73"
PAPER = "#FFFFFF"
CANVAS = "#F5F5F7"
LINE = "#E5E5EA"

PURPLE = "#7C3AED"
GREEN = "#34C759"
ORANGE = "#FF9500"
RED = "#FF3B30"
ROSE = "#FF2D55"
BLACK = "#0A0A0B"

MODEL_COLORS = {"GCN": PURPLE, "GraphSAGE": GREEN, "GAT": ORANGE}


def inject_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont,
                'SF Pro Display', 'Segoe UI', Roboto, sans-serif;
        }}

        .stApp {{
            background-color: {CANVAS};
        }}

        section[data-testid="stSidebar"] {{
            background-color: {BLACK};
        }}
        section[data-testid="stSidebar"] * {{
            color: #F5F5F7 !important;
        }}
        section[data-testid="stSidebar"] .stRadio label {{
            font-size: 0.95rem;
            padding: 4px 0;
        }}

        h1, h2, h3 {{
            color: {INK};
            font-weight: 700;
            letter-spacing: -0.02em;
        }}
        p, li, span {{
            color: {INK};
        }}
        .subtle {{
            color: {INK_SOFT};
            font-size: 0.92rem;
        }}

        .card {{
            background: {PAPER};
            border: 1px solid {LINE};
            border-radius: 18px;
            padding: 24px 26px;
            box-shadow: 0 2px 14px rgba(0,0,0,0.04);
            margin-bottom: 20px;
        }}

        .stat-value {{
            font-size: 2.1rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            color: {INK};
            line-height: 1.1;
        }}
        .stat-label {{
            font-size: 0.85rem;
            color: {INK_SOFT};
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-top: 4px;
        }}

        .badge {{
            display: inline-block;
            padding: 3px 12px;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }}
        .badge-bonus {{ background: {ROSE}22; color: {ROSE}; }}
        .badge-required {{ background: {PURPLE}22; color: {PURPLE}; }}
        .badge-good {{ background: {GREEN}22; color: {GREEN}; }}
        .badge-warn {{ background: {ORANGE}22; color: {ORANGE}; }}
        .badge-low {{ background: {RED}22; color: {RED}; }}

        .hero {{
            background: linear-gradient(135deg, {BLACK} 0%, #1D1233 100%);
            border-radius: 24px;
            padding: 40px 44px;
            margin-bottom: 24px;
        }}
        .hero h1 {{ color: {PAPER}; margin-bottom: 6px; }}
        .hero p {{ color: #C9C9CE; font-size: 1.05rem; max-width: 680px; }}

        .explain-box {{
            background: {PURPLE}0D;
            border-left: 3px solid {PURPLE};
            border-radius: 10px;
            padding: 14px 18px;
            font-size: 0.92rem;
            color: {INK};
            margin-top: 10px;
        }}

        div[data-baseweb="select"] > div {{
            border-radius: 12px !important;
        }}

        .stButton button {{
            background: {PURPLE};
            color: white;
            border-radius: 12px;
            border: none;
            font-weight: 600;
            padding: 10px 22px;
        }}
        .stButton button:hover {{
            background: #6D28D9;
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def stat_card(value: str, label: str, accent: str = INK):
    st.markdown(
        f"""
        <div class="card">
            <div class="stat-value" style="color:{accent}">{value}</div>
            <div class="stat-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def explain(text: str):
    st.markdown(f'<div class="explain-box">{text}</div>', unsafe_allow_html=True)


def badge(text: str, kind: str = "required"):
    st.markdown(f'<span class="badge badge-{kind}">{text}</span>', unsafe_allow_html=True)


def show_image(path: str):
    """Version-safe image display — older Streamlit builds don't
    support the use_container_width argument."""
    try:
        st.image(path, use_container_width=True)
    except TypeError:
        st.image(path, use_column_width=True)


def style_fig(fig):
    """
    Applies consistent, explicit light-theme styling to every Plotly
    chart in the dashboard. Plotly can otherwise pick a dark-mode-
    oriented default template that renders axis labels and titles in
    very light gray — nearly invisible against this dashboard's white
    background — so every color here is set explicitly rather than
    left to Plotly's auto-detection.
    """
    fig.update_layout(
        plot_bgcolor=PAPER,
        paper_bgcolor=PAPER,
        font=dict(color=INK, family="Inter, -apple-system, sans-serif", size=13),
        legend=dict(font=dict(color=INK)),
        title=dict(font=dict(color=INK)),
        margin=dict(t=30, b=40, l=10, r=10),
    )
    fig.update_xaxes(
        color=INK,
        title_font=dict(color=INK),
        tickfont=dict(color=INK_SOFT),
        gridcolor=LINE,
        linecolor=LINE,
        zerolinecolor=LINE,
    )
    fig.update_yaxes(
        color=INK,
        title_font=dict(color=INK),
        tickfont=dict(color=INK_SOFT),
        gridcolor=LINE,
        linecolor=LINE,
        zerolinecolor=LINE,
    )
    return fig


# ============================================================
# DATA / MODEL LOADING (cached, defensive — never crash the UI)
# ============================================================

def find_project_root(marker: str = "requirements.txt") -> Path:
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent
    return Path.cwd()


PROJECT_ROOT = find_project_root()
sys.path.append(str(PROJECT_ROOT / "src"))


@st.cache_data(show_spinner=False)
def load_processed_data():
    path = PROJECT_ROOT / "data" / "processed" / "processed_data.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_model_config():
    path = PROJECT_ROOT / "models_checkpoints" / "model_config.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_csv(relative_path: str):
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def load_trained_models(_config):
    """Loads GCN, GraphSAGE, GAT with weights, defensively correcting
    hidden_channels against each checkpoint's real shape in case the
    config file is stale relative to what was actually trained."""
    if _config is None:
        return {}

    try:
        from models.gnn_models import GCN, GraphSAGE, GAT
    except Exception:
        return {}

    checkpoint_dir = PROJECT_ROOT / "models_checkpoints"
    device = torch.device("cpu")

    def build_and_load(model_class, config_entry, checkpoint_path, shape_key=None):
        if not checkpoint_path.exists():
            return None
        state_dict = torch.load(checkpoint_path, map_location=device)
        cfg = {k: v for k, v in config_entry.items() if k != "class"}
        if shape_key is not None and shape_key[0] in state_dict:
            actual = state_dict[shape_key[0]].shape[shape_key[1]]
            if actual != cfg.get("hidden_channels"):
                cfg["hidden_channels"] = actual
        model = model_class(**cfg).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    models = {}
    if "gcn" in _config:
        models["GCN"] = build_and_load(
            GCN, _config["gcn"], checkpoint_dir / "gcn_final.pt", ("bn1.weight", 0)
        )
    if "graphsage" in _config:
        models["GraphSAGE"] = build_and_load(
            GraphSAGE, _config["graphsage"], checkpoint_dir / "graphsage_final.pt", ("bn1.weight", 0)
        )
    if "gat" in _config:
        models["GAT"] = build_and_load(
            GAT, _config["gat"], checkpoint_dir / "gat_final.pt", ("conv1.att_src", 2)
        )
    return {k: v for k, v in models.items() if v is not None}


ARXIV_CATEGORIES = {
    0: "Numerical Analysis", 1: "Multiagent Systems", 2: "Logic in CS",
    3: "Computation and Language", 4: "Machine Learning", 5: "Cryptography and Security",
    6: "Distributed Computing", 7: "Human-Computer Interaction", 8: "Computer Vision",
    9: "Computational Geometry", 10: "Artificial Intelligence", 11: "Information Retrieval",
    12: "Networking", 13: "Software Engineering", 14: "Robotics", 15: "Databases",
    16: "Data Structures & Algorithms", 17: "Systems and Control", 18: "Neural and Evolutionary Computing",
    19: "Graphics", 20: "Computational Complexity", 21: "Formal Languages",
    22: "Operating Systems", 23: "Information Theory", 24: "Programming Languages",
    25: "Social and Information Networks", 26: "Discrete Mathematics",
    27: "Symbolic Computation", 28: "Hardware Architecture", 29: "Performance",
    30: "Emerging Technologies", 31: "General Literature", 32: "Digital Libraries",
    33: "Computers and Society", 34: "Sound", 35: "Multimedia",
    36: "Mathematical Software", 37: "Numerical & Scientific Computing",
    38: "Statistics Theory", 39: "Other",
}


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

inject_css()

with st.sidebar:
    st.markdown(
        f"<h2 style='color:white; margin-bottom:0;'>◆ Graph Intelligence</h2>"
        f"<p style='color:#9A9AA0; font-size:0.85rem; margin-top:2px;'>OGBN-Arxiv · CCS4354</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border-color:#333;'>", unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Graph Statistics",
            "Model Performance",
            "Classify a Paper",
            "Embedding Explorer",
            "Explainability",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#333;'>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#9A9AA0; font-size:0.78rem;'>Models: GCN · GraphSAGE · "
        "<span style='color:#FF2D55;'>GAT (bonus)</span></p>",
        unsafe_allow_html=True,
    )


# ============================================================
# LOAD EVERYTHING ONCE
# ============================================================

processed = load_processed_data()
model_config = load_model_config()
eval_results = load_csv("reports/model_evaluation_results.csv")
models = load_trained_models(model_config)

DATA_READY = processed is not None
MODELS_READY = len(models) > 0


# ============================================================
# PAGE: OVERVIEW
# ============================================================

def render_overview():
    st.markdown(
        f"""
        <div class="hero">
            <h1>Graph Intelligence Dashboard</h1>
            <p>Explore how three graph neural networks classify research papers
            in the OGBN-Arxiv citation network — by their own content, and by
            who cites and is cited by whom.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not DATA_READY:
        st.warning(
            "Processed data not found. Place `processed_data.pkl` in "
            "`data/processed/` to activate live statistics."
        )
        return

    features = processed["features"]
    edge_index = processed["edge_index"]
    labels = processed["labels"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        stat_card(f"{features.shape[0]:,}", "Papers (nodes)", PURPLE)
    with c2:
        stat_card(f"{edge_index.shape[1]:,}", "Citation links (edges)", GREEN)
    with c3:
        stat_card(f"{int(labels.max().item()) + 1}", "Subject categories", ORANGE)
    with c4:
        stat_card(f"{features.shape[1]}", "Features per paper", ROSE)

    st.markdown("### How to read this dashboard")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(
            """<div class="card">
            <b>◆ Graph Statistics</b>
            <p class="subtle">Structural properties of the citation network itself —
            how densely connected it is, how citation counts are distributed,
            and whether papers form one connected community or many isolated ones.</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            """<div class="card">
            <b>◆ Model Performance</b>
            <p class="subtle">How accurately each of the three models classifies
            papers it has never seen before, and where each model's strengths
            and weaknesses lie.</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_c:
        st.markdown(
            """<div class="card">
            <b>◆ Classify a Paper</b>
            <p class="subtle">Pick any real paper from the dataset, choose a
            model, and see its prediction with a plain-language explanation
            of how confident it is and why.</p>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("### The three models, at a glance")
    m1, m2, m3 = st.columns(3)
    model_info = [
        ("GCN", PURPLE, "required",
         "Aggregates each paper's information with its direct neighborhood using a fixed, "
         "mathematically-normalized rule. Fast and stable, treats every neighboring "
         "paper's influence equally."),
        ("GraphSAGE", GREEN, "required",
         "Learns how to aggregate neighboring papers rather than using a fixed rule. "
         "Generalizes well and scales to large citation networks."),
        ("GAT", ORANGE, "bonus",
         "Learns an individual attention weight for every citation link — meaning it "
         "decides which neighboring papers matter more for a given prediction, "
         "rather than treating them all the same."),
    ]
    for col, (name, color, kind, desc) in zip([m1, m2, m3], model_info):
        with col:
            st.markdown(
                f"""
                <div class="card" style="border-top: 3px solid {color};">
                    <span class="badge badge-{'bonus' if kind=='bonus' else 'required'}">
                        {'Bonus architecture' if kind=='bonus' else 'Required architecture'}
                    </span>
                    <h3 style="margin-top:10px;">{name}</h3>
                    <p class="subtle">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# PAGE: GRAPH STATISTICS
# ============================================================

def render_graph_statistics():
    st.markdown("## Graph Statistics")
    st.markdown(
        '<p class="subtle">Structural properties of the OGBN-Arxiv citation network, '
        "computed directly from the processed dataset.</p>",
        unsafe_allow_html=True,
    )

    if not DATA_READY:
        st.warning("Processed data not found — cannot compute statistics.")
        return

    features = processed["features"]
    edge_index = processed["edge_index"]
    labels = processed["labels"].view(-1).numpy()

    num_nodes = features.shape[0]
    num_edges = edge_index.shape[1]

    deg = np.bincount(edge_index[0].numpy(), minlength=num_nodes)
    density = num_edges / (num_nodes * (num_nodes - 1))

    c1, c2, c3 = st.columns(3)
    with c1:
        stat_card(f"{deg.mean():.2f}", "Average citations per paper", PURPLE)
    with c2:
        stat_card(f"{deg.max():,}", "Most-cited paper's links", GREEN)
    with c3:
        stat_card(f"{density:.2e}", "Graph density", ORANGE)

    explain(
        "<b>What this means:</b> a density this low is expected for a real citation "
        "network — most papers cite only a small handful of others out of 169,000+ "
        "possible papers. This sparsity is exactly why graph neural networks (which "
        "exploit the specific structure rather than assuming a dense network) are a "
        "good fit for this data."
    )

    st.markdown("### Degree distribution")
    fig = px.histogram(
        x=deg, nbins=60, log_y=True,
        labels={"x": "Number of citation links", "y": "Number of papers"},
        color_discrete_sequence=[PURPLE],
    )
    fig.update_layout(bargap=0.05)
    style_fig(fig)
    st.plotly_chart(fig, use_container_width=True)
    explain(
        "Most papers have very few citation links, while a small number are cited "
        "extremely often — a typical <b>power-law pattern</b> seen in real-world "
        "citation and social networks."
    )

    st.markdown("### Papers per subject category")
    cat_counts = pd.Series(labels).value_counts().sort_index()
    cat_df = pd.DataFrame({
        "Category": [ARXIV_CATEGORIES.get(i, f"Class {i}") for i in cat_counts.index],
        "Papers": cat_counts.values,
    }).sort_values("Papers", ascending=True).tail(15)
    fig2 = px.bar(
        cat_df, x="Papers", y="Category", orientation="h",
        color_discrete_sequence=[GREEN],
    )
    style_fig(fig2)
    st.plotly_chart(fig2, use_container_width=True)
    explain(
        "Category sizes are uneven — some subject areas (like Machine Learning) have "
        "far more papers than others. This <b>class imbalance</b> is one reason "
        "precision and recall are reported per-model alongside accuracy, since "
        "accuracy alone can hide poor performance on smaller categories."
    )


# ============================================================
# PAGE: MODEL PERFORMANCE
# ============================================================

def render_model_performance():
    st.markdown("## Model Performance")
    st.markdown(
        '<p class="subtle">Accuracy, precision, recall, and F1-score for each model, '
        "measured on data the model never saw during training.</p>",
        unsafe_allow_html=True,
    )

    if eval_results is None:
        st.warning(
            "Evaluation results not found. Expecting "
            "`reports/model_evaluation_results.csv`."
        )
        return

    test_df = eval_results[eval_results["Dataset"] == "Test"].set_index("Model")

    cols = st.columns(len(test_df))
    for col, model_name in zip(cols, test_df.index):
        color = MODEL_COLORS.get(model_name, INK)
        acc = test_df.loc[model_name, "Accuracy"]
        with col:
            kind = "bonus" if model_name == "GAT" else "required"
            st.markdown(
                f"""
                <div class="card" style="border-top:3px solid {color};">
                    <span class="badge badge-{kind}">{'Bonus' if kind=='bonus' else 'Required'}</span>
                    <div class="stat-value" style="color:{color}; margin-top:8px;">{acc:.1%}</div>
                    <div class="stat-label">{model_name} test accuracy</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Full comparison")
    metric_cols = ["Accuracy", "Precision", "Recall", "F1 Score"]
    fig = go.Figure()
    for model_name in test_df.index:
        fig.add_trace(go.Bar(
            name=model_name,
            x=metric_cols,
            y=[test_df.loc[model_name, m] for m in metric_cols],
            marker_color=MODEL_COLORS.get(model_name, INK),
        ))
    fig.update_layout(
        barmode="group",
        yaxis_tickformat=".0%",
        legend_title_text="Model",
    )
    style_fig(fig)
    st.plotly_chart(fig, use_container_width=True)

    best_acc_model = test_df["Accuracy"].idxmax()
    best_f1_model = test_df["F1 Score"].idxmax()
    explain(
        f"<b>{best_acc_model}</b> achieves the highest overall test accuracy. "
        f"<b>{best_f1_model}</b> achieves the highest macro F1-score — meaning it "
        "performs most consistently <i>across all 40 subject categories</i>, not just "
        "the largest ones. A model can have high accuracy while still doing poorly on "
        "rarer categories, which is why both metrics are shown rather than accuracy alone."
    )

    st.markdown("### Validation vs. test — why both are shown")
    val_df = eval_results[eval_results["Dataset"] == "Validation"].set_index("Model")
    compare = pd.DataFrame({
        "Validation Accuracy": val_df["Accuracy"],
        "Test Accuracy": test_df["Accuracy"],
    })
    st.dataframe(compare.style.format("{:.2%}"), use_container_width=True)
    explain(
        "OGBN-Arxiv splits papers <b>by publication year</b> — older papers for "
        "training and validation, newer papers for testing. A drop from validation "
        "to test accuracy is expected and reflects a realistic challenge: predicting "
        "categories for genuinely newer research the model has never encountered."
    )


# ============================================================
# PAGE: CLASSIFY A PAPER (live prediction)
# ============================================================

def render_prediction():
    st.markdown("## Classify a Paper")
    st.markdown(
        '<p class="subtle">Pick a real paper from the test set and see how each '
        "model classifies it — along with what the model actually used to decide.</p>",
        unsafe_allow_html=True,
    )

    if not (DATA_READY and MODELS_READY):
        st.warning(
            "Trained models or processed data not found. This page needs "
            "`processed_data.pkl` and the `.pt` checkpoint files in "
            "`models_checkpoints/`."
        )
        return

    features = processed["features"]
    edge_index = processed["edge_index"]
    labels = processed["labels"].view(-1)
    test_idx = processed["test_idx"].view(-1).numpy()

    col1, col2 = st.columns([2, 1])
    with col1:
        node_choice = st.selectbox(
            "Choose a paper (by node ID)",
            options=list(test_idx[:200]),
            format_func=lambda i: f"Paper #{i}",
        )
    with col2:
        model_choice = st.selectbox("Choose a model", options=list(models.keys()))

    if st.button("Classify this paper"):
        model = models[model_choice]
        with torch.no_grad():
            logits = model(features.float(), edge_index.long())
            probs = torch.softmax(logits[node_choice], dim=0)

        predicted_class = int(probs.argmax().item())
        confidence = float(probs[predicted_class].item())
        true_class = int(labels[node_choice].item())

        color = MODEL_COLORS.get(model_choice, INK)
        conf_kind = "good" if confidence > 0.7 else ("warn" if confidence > 0.4 else "low")

        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(
                f"""<div class="card">
                <div class="stat-label">Predicted category</div>
                <div class="stat-value" style="color:{color}; font-size:1.4rem;">
                    {ARXIV_CATEGORIES.get(predicted_class, predicted_class)}
                </div>
                </div>""",
                unsafe_allow_html=True,
            )
        with r2:
            st.markdown(
                f"""<div class="card">
                <span class="badge badge-{conf_kind}">{confidence:.1%} confidence</span>
                <div class="stat-label" style="margin-top:10px;">Model's certainty</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with r3:
            match = "✓ Matches actual label" if predicted_class == true_class else "✗ Differs from actual label"
            match_color = GREEN if predicted_class == true_class else RED
            st.markdown(
                f"""<div class="card">
                <div class="stat-value" style="color:{match_color}; font-size:1.1rem;">{match}</div>
                <div class="stat-label">Actual: {ARXIV_CATEGORIES.get(true_class, true_class)}</div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("### What the model used to decide")
        neighbor_count = int(((edge_index[0] == node_choice) | (edge_index[1] == node_choice)).sum())
        explain(
            f"<b>{model_choice}</b> based this prediction on two sources of information: "
            f"(1) this paper's own 128-dimensional content features, derived from its title "
            f"and abstract, and (2) its position in the citation graph — it has "
            f"<b>{neighbor_count} citation links</b> to other papers, and "
            f"{model_choice} aggregated information from those neighboring papers when "
            "forming its prediction. "
            + ("Because GAT uses attention, some of those neighbors were weighted more "
               "heavily than others — see the Explainability page for a breakdown."
               if model_choice == "GAT" else
               "GCN and GraphSAGE both consider every neighbor, though GraphSAGE learns "
               "how to combine them rather than using a fixed rule.")
        )

        st.markdown("### Top 5 predicted categories")
        top5_idx = torch.topk(probs, 5).indices.numpy()
        top5_df = pd.DataFrame({
            "Category": [ARXIV_CATEGORIES.get(int(i), int(i)) for i in top5_idx],
            "Probability": [float(probs[i]) for i in top5_idx],
        })
        fig = px.bar(
            top5_df, x="Probability", y="Category", orientation="h",
            color_discrete_sequence=[color],
        )
        fig.update_layout(xaxis_tickformat=".0%")
        style_fig(fig)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE: EMBEDDING EXPLORER
# ============================================================

def render_embeddings():
    st.markdown("## Embedding Explorer")
    st.markdown(
        '<p class="subtle">A 2D snapshot of how each model internally represents '
        "papers — points close together are papers the model considers similar.</p>",
        unsafe_allow_html=True,
    )

    img_path = PROJECT_ROOT / "reports" / "embedding_visualization.png"
    if img_path.exists():
        show_image(str(img_path))
        explain(
            "Each column is one model's learned representation, reduced to 2 dimensions "
            "using <b>PCA</b> (top row, fast and linear) and <b>t-SNE</b> (bottom row, "
            "slower but often reveals cluster structure more clearly). Colors indicate "
            "the paper's true subject category — <b>tighter, more separated clusters</b> "
            "generally indicate a model that has learned more discriminative representations."
        )
    else:
        st.warning(
            "Embedding visualization not found. Run Task 07's notebook to generate "
            "`reports/embedding_visualization.png`."
        )


# ============================================================
# PAGE: EXPLAINABILITY
# ============================================================

def render_explainability():
    st.markdown("## Explainability")
    st.markdown(
        '<p class="subtle">How GAT\'s attention mechanism and other techniques reveal '
        "which parts of the graph and which input features actually drive a prediction.</p>",
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["Attention Weights", "Feature Importance", "Neighborhood Influence"])

    with tab1:
        img_path = PROJECT_ROOT / "reports" / "gat_attention_weights.png"
        if img_path.exists():
            show_image(str(img_path))
            explain(
                "GAT assigns a learned <b>attention weight</b> to every citation link. "
                "This chart shows the most heavily-weighted neighbors for one example "
                "paper — a wider spread of weights means GAT is genuinely discriminating "
                "between more and less relevant neighbors, rather than treating them equally."
            )
        else:
            st.info("Attention weight chart not yet generated.")

    with tab2:
        img_path = PROJECT_ROOT / "reports" / "feature_importance.png"
        if img_path.exists():
            show_image(str(img_path))
            explain(
                "<b>Bonus technique.</b> Each input feature is temporarily zeroed out, "
                "and the drop in the model's prediction confidence is measured. Features "
                "with a larger confidence drop were more influential in the original prediction."
            )
        else:
            st.info("Feature importance chart not yet generated.")

    with tab3:
        img_path = PROJECT_ROOT / "reports" / "neighborhood_influence.png"
        if img_path.exists():
            show_image(str(img_path))
            explain(
                "<b>Bonus technique.</b> Compares a paper's prediction with and without "
                "its citation links. A large difference means the model relies heavily on "
                "graph structure, not just the paper's own content, to classify it."
            )
        else:
            st.info("Neighborhood influence chart not yet generated.")


# ============================================================
# ROUTER
# ============================================================

PAGES = {
    "Overview": render_overview,
    "Graph Statistics": render_graph_statistics,
    "Model Performance": render_model_performance,
    "Classify a Paper": render_prediction,
    "Embedding Explorer": render_embeddings,
    "Explainability": render_explainability,
}

PAGES[page]()