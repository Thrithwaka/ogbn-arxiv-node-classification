"""
Task 08 - Graph Intelligence Dashboard
CCS4354 - Tensors and Graphs | OGBN-Arxiv

A Streamlit dashboard showing:
  - Graph statistics
  - Model performance metrics (GCN vs GAT)
  - Node classification results (look up any node)
  - Embedding visualizations (PCA / t-SNE)

Run with:
    streamlit run dashboard/app.py

Expects the processed dataset under data/processed/ and trained model
checkpoints under models_checkpoints/.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "processed_data.pkl"
GCN_CKPT_PATH = PROJECT_ROOT / "models_checkpoints" / "gcn_final.pt"
GAT_CKPT_PATH = PROJECT_ROOT / "models_checkpoints" / "gat_final.pt"

st.set_page_config(
    page_title="Arxiv Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

    :root {
        --ink: #27233f;
        --muted: #8a86a3;
        --purple: #6e59bd;
        --purple-dark: #51409b;
        --lavender: #eeeafe;
        --pink: #f36f9d;
        --surface: #ffffff;
        --canvas: #e8ecf6;
    }

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: var(--canvas); color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { right: 1.25rem; }
    [data-testid="stAppViewContainer"] > .main {
        background: var(--surface);
        border-radius: 28px 0 0 28px;
        margin: 18px 18px 18px 0;
        box-shadow: 0 20px 55px rgba(55, 47, 102, .12);
        height: calc(100vh - 36px);
        overflow-x: hidden;
        overflow-y: auto;
        scrollbar-gutter: stable;
    }
    .main .block-container {
        max-width: 1440px;
        padding: 2.25rem 3rem 4rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #7562c8 0%, #5b49ad 100%);
        border-radius: 28px;
        margin: 18px 0 18px 18px;
        min-width: 245px;
        box-shadow: 0 20px 45px rgba(71, 58, 143, .24);
    }
    [data-testid="stSidebar"] > div:first-child { padding: 1.6rem 1rem; }
    [data-testid="stSidebar"] > div:first-child { overflow-y: auto; }
    [data-testid="stSidebar"] * { color: #fff; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: rgba(255,255,255,.72);
    }
    [data-testid="stSidebar"] [role="radiogroup"] { gap: .45rem; }
    [data-testid="stSidebar"] label {
        padding: .7rem .75rem;
        border-radius: 12px;
        transition: background .2s ease, transform .2s ease;
    }
    [data-testid="stSidebar"] label:hover {
        background: rgba(255,255,255,.12);
        transform: translateX(2px);
    }
    [data-testid="stSidebar"] label:has(input:checked) {
        background: rgba(255,255,255,.2);
        box-shadow: inset 0 0 0 1px rgba(255,255,255,.12);
    }

    .dashboard-kicker {
        color: var(--purple); font-size: .72rem; font-weight: 700;
        letter-spacing: .14em; text-transform: uppercase; margin-bottom: .25rem;
    }
    .dashboard-title {
        color: var(--ink); font-size: clamp(2rem, 4vw, 3.35rem); line-height: 1.04;
        font-weight: 700; letter-spacing: -.055em; margin: 0;
    }
    .dashboard-subtitle {
        color: var(--muted); font-size: 1rem; margin: .75rem 0 1.8rem;
    }
    .hero-card {
        position: relative; overflow: hidden; padding: 1.75rem 2rem;
        border-radius: 24px; color: white;
        background: linear-gradient(135deg, #5845a6 0%, #7763c9 62%, #8b74db 100%);
        box-shadow: 0 16px 36px rgba(83, 65, 166, .22);
        margin-bottom: 1.7rem;
    }
    .hero-card:after {
        content: ''; position: absolute; width: 310px; height: 310px;
        right: -85px; top: -190px; border-radius: 50%;
        border: 45px solid rgba(255,255,255,.07);
    }
    .hero-eyebrow { font-size: .75rem; opacity: .72; text-transform: uppercase; letter-spacing: .12em; }
    .hero-card h2 { color: white; font-size: 1.55rem; margin: .35rem 0 .3rem; }
    .hero-card p { margin: 0; opacity: .78; max-width: 670px; }
    .hero-pill {
        display: inline-block; margin-top: 1rem; padding: .36rem .7rem;
        border: 1px solid rgba(255,255,255,.2); border-radius: 999px;
        background: rgba(255,255,255,.1); font-size: .76rem;
    }

    h1, h2, h3 { color: var(--ink); letter-spacing: -.025em; }
    h2 { padding-top: .2rem; }
    [data-testid="stMetric"] {
        background: #fff; border: 1px solid #efedf7; border-radius: 18px;
        padding: 1.05rem 1.15rem; min-height: 112px;
        box-shadow: 0 9px 25px rgba(58, 48, 105, .07);
    }
    [data-testid="stMetricLabel"] { color: var(--muted); font-weight: 600; }
    [data-testid="stMetricValue"] {
        color: var(--ink); font-weight: 700; font-size: clamp(1.45rem, 2.1vw, 2rem);
        letter-spacing: -.035em; white-space: nowrap;
    }
    [data-testid="stMetricDelta"] { color: var(--pink); }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid #efedf7; border-radius: 18px; overflow: hidden;
        box-shadow: 0 9px 25px rgba(58, 48, 105, .06);
    }
    .table-shell {
        width: 100%; overflow-x: auto; background: #fff;
        border: 1px solid #ece8f6; border-radius: 18px;
        box-shadow: 0 9px 25px rgba(58, 48, 105, .07);
    }
    .dashboard-table {
        width: 100%; min-width: 480px; border-collapse: collapse;
        color: var(--ink); font-size: .94rem;
    }
    .dashboard-table thead th {
        background: linear-gradient(135deg, #6753b8, #7d68cc);
        color: #fff; font-size: .76rem; font-weight: 700;
        letter-spacing: .07em; text-transform: uppercase;
        padding: .9rem 1.1rem; text-align: left;
        border: 0;
    }
    .dashboard-table tbody td {
        background: #fff; color: var(--ink); padding: .9rem 1.1rem;
        border: 0; border-bottom: 1px solid #efedf7;
    }
    .dashboard-table tbody tr:nth-child(even) td { background: #faf9fe; }
    .dashboard-table tbody tr:hover td { background: #f1edfd; }
    .dashboard-table tbody tr:last-child td { border-bottom: 0; }
    .dashboard-table th:not(:first-child),
    .dashboard-table td:not(:first-child) { text-align: right; }
    .dashboard-table tbody td:first-child { font-weight: 600; color: #51409b; }
    [data-testid="stImage"] {
        background: #fff; border: 1px solid #efedf7; border-radius: 18px;
        padding: .55rem; overflow: hidden;
        box-shadow: 0 9px 25px rgba(58, 48, 105, .06);
    }
    .stButton > button {
        background: linear-gradient(135deg, var(--purple), var(--purple-dark));
        color: white; border: 0; border-radius: 12px; padding: .6rem 1rem;
        box-shadow: 0 8px 18px rgba(87, 68, 166, .2);
    }
    .stButton > button:hover { color: white; border: 0; transform: translateY(-1px); }
    div[data-baseweb="select"] > div, .stNumberInput input, .stTextArea textarea {
        background: #fbfaff !important; border-radius: 12px !important;
        border-color: #e7e3f2 !important;
    }
    [data-testid="stAppViewContainer"] > .main [data-testid="stWidgetLabel"] p,
    [data-testid="stAppViewContainer"] > .main [data-testid="stSlider"] p,
    [data-testid="stAppViewContainer"] > .main [data-testid="stRadio"] p,
    [data-testid="stAppViewContainer"] > .main [data-testid="stSelectbox"] p,
    [data-testid="stAppViewContainer"] > .main [data-testid="stNumberInput"] p,
    [data-testid="stAppViewContainer"] > .main [data-testid="stTextArea"] p {
        color: var(--ink) !important; font-weight: 500;
    }
    [data-testid="stAppViewContainer"] > .main [data-testid="stRadio"] > div[role="radiogroup"] {
        background: #f5f2fd; border-radius: 14px; padding: .35rem;
        width: fit-content;
    }
    [data-testid="stAppViewContainer"] > .main [data-testid="stRadio"] > div[role="radiogroup"] label {
        padding: .35rem .65rem; border-radius: 10px;
    }
    [data-testid="stSlider"] [role="slider"] { background: var(--purple); }
    [data-testid="stAlert"] { border-radius: 14px; }
    ::-webkit-scrollbar { width: 9px; height: 9px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #c9c1e7; border-radius: 999px; }
    ::-webkit-scrollbar-thumb:hover { background: #a99bd8; }
    hr { border-color: #efedf7; }

    @media (max-width: 900px) {
        [data-testid="stAppViewContainer"] > .main {
            margin: 8px; border-radius: 20px; height: calc(100vh - 16px);
        }
        [data-testid="stSidebar"] { margin: 8px; border-radius: 20px; }
        .main .block-container { padding: 1.5rem 1.15rem 3rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# MODEL DEFINITIONS (must match the checkpoints exactly)
# ============================================================

class GCN(torch.nn.Module):
    def __init__(self, in_channels=128, hidden_channels=256, out_channels=40, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, return_embedding=False):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        embedding = x
        x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.conv3(x, edge_index)
        if return_embedding:
            return out, embedding
        return out


class GAT(torch.nn.Module):
    def __init__(self, in_channels=128, hidden_channels=64, out_channels=40, heads=4, dropout=0.5):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels * heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index, return_embedding=False):
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.elu(x1)
        embedding = x1
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        out = self.conv2(x1, edge_index)
        if return_embedding:
            return out, embedding
        return out


# ============================================================
# DATA / MODEL LOADING (cached so the dashboard stays fast)
# ============================================================

@st.cache_resource
def load_data():
    if not DATASET_PATH.is_file():
        return None, None, None, None

    with open(DATASET_PATH, "rb") as f:
        dataset = pickle.load(f)

    x = torch.as_tensor(dataset["features"], dtype=torch.float)
    y = torch.as_tensor(dataset["labels"])
    edge_index = torch.as_tensor(dataset["edge_index"], dtype=torch.long)
    train_idx = torch.as_tensor(dataset["train_idx"], dtype=torch.long)
    valid_idx = torch.as_tensor(dataset["valid_idx"], dtype=torch.long)
    test_idx = torch.as_tensor(dataset["test_idx"], dtype=torch.long)

    if y.dim() > 1 and y.shape[-1] == 1:
        y = y.squeeze(-1)
    if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
        edge_index = edge_index.t()

    data = Data(x=x, edge_index=edge_index, y=y).to(DEVICE)
    train_idx, valid_idx, test_idx = train_idx.to(DEVICE), valid_idx.to(DEVICE), test_idx.to(DEVICE)
    return data, train_idx, valid_idx, test_idx


@st.cache_resource
def load_models(in_channels, num_classes):
    if not (GCN_CKPT_PATH.is_file() and GAT_CKPT_PATH.is_file()):
        return None, None

    gcn_model = GCN(in_channels=in_channels, hidden_channels=256, out_channels=num_classes).to(DEVICE)
    gat_model = GAT(in_channels=in_channels, hidden_channels=64, out_channels=num_classes, heads=4).to(DEVICE)

    gcn_state = torch.load(GCN_CKPT_PATH, map_location=DEVICE, weights_only=False)
    gat_state = torch.load(GAT_CKPT_PATH, map_location=DEVICE, weights_only=False)
    if isinstance(gcn_state, dict) and "state_dict" in gcn_state:
        gcn_state = gcn_state["state_dict"]
    if isinstance(gat_state, dict) and "state_dict" in gat_state:
        gat_state = gat_state["state_dict"]

    gcn_model.load_state_dict(gcn_state)
    gat_model.load_state_dict(gat_state)
    gcn_model.eval()
    gat_model.eval()
    return gcn_model, gat_model


@st.cache_data
def compute_graph_stats(_data):
    num_nodes = _data.num_nodes
    edge_index = _data.edge_index.cpu().numpy()
    src, dst = edge_index[0], edge_index[1]

    adj = csr_matrix((np.ones(len(src)), (src, dst)), shape=(num_nodes, num_nodes))
    num_edges = adj.nnz

    out_degree = np.array(adj.sum(axis=1)).flatten()
    in_degree = np.array(adj.sum(axis=0)).flatten()
    total_degree = out_degree + in_degree

    density = num_edges / (num_nodes * (num_nodes - 1))

    n_components, comp_labels = connected_components(adj, directed=True, connection="weak")
    comp_sizes = np.bincount(comp_labels)
    largest_component_frac = comp_sizes.max() / num_nodes

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "density": density,
        "avg_degree": total_degree.mean(),
        "median_degree": np.median(total_degree),
        "max_degree": int(total_degree.max()),
        "min_degree": int(total_degree.min()),
        "num_components": int(n_components),
        "largest_component_frac": largest_component_frac,
        "degree_array": total_degree,
    }


@st.cache_data
def compute_predictions(_gcn_model, _gat_model, _data):
    with torch.no_grad():
        gcn_logits = _gcn_model(_data.x, _data.edge_index)
        gat_logits = _gat_model(_data.x, _data.edge_index)
    gcn_probs = F.softmax(gcn_logits, dim=-1).cpu().numpy()
    gat_probs = F.softmax(gat_logits, dim=-1).cpu().numpy()
    return gcn_probs, gat_probs


@st.cache_data
def compute_embeddings(_gcn_model, _gat_model, _data, sample_size=5000):
    with torch.no_grad():
        _, gcn_emb = _gcn_model(_data.x, _data.edge_index, return_embedding=True)
        _, gat_emb = _gat_model(_data.x, _data.edge_index, return_embedding=True)
    gcn_emb = gcn_emb.cpu().numpy()
    gat_emb = gat_emb.cpu().numpy()

    n = gcn_emb.shape[0]
    sample_size = min(sample_size, n)
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(n, size=sample_size, replace=False)
    return gcn_emb, gat_emb, sample_idx


def metrics_for(y_true, y_pred, idx):
    y_t = y_true[idx]
    y_p = y_pred[idx]
    return {
        "Accuracy": accuracy_score(y_t, y_p),
        "Precision (weighted)": precision_score(y_t, y_p, average="weighted", zero_division=0),
        "Recall (weighted)": recall_score(y_t, y_p, average="weighted", zero_division=0),
        "F1 (weighted)": f1_score(y_t, y_p, average="weighted", zero_division=0),
    }


def style_chart(fig, ax):
    """Apply the dashboard visual system to Matplotlib charts."""
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfaff")
    ax.grid(axis="y", color="#e9e5f5", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d8d1ec")
    ax.spines["bottom"].set_color("#d8d1ec")
    ax.tick_params(colors="#77718f")
    ax.xaxis.label.set_color("#68627e")
    ax.yaxis.label.set_color("#68627e")
    ax.title.set_color("#27233f")
    # Avoid tight_layout here: Matplotlib can route logarithmic tick labels
    # through its math-text parser and fail while calculating their bounds.
    # Fixed margins are predictable across the histogram, bars and scatter plots.
    fig.subplots_adjust(left=0.11, right=0.96, bottom=0.18, top=0.88)
    return fig


def render_summary_table(frame):
    """Render small read-only results without Streamlit's theme-dependent grid."""
    table_html = frame.to_html(index=False, classes="dashboard-table", border=0, escape=True)
    st.markdown(f'<div class="table-shell">{table_html}</div>', unsafe_allow_html=True)


# ============================================================
# APP LAYOUT
# ============================================================

st.markdown(
    """
    <div class="dashboard-kicker">Graph learning workspace</div>
    <h1 class="dashboard-title">Arxiv Intelligence</h1>
    <p class="dashboard-subtitle">Explore the citation network, compare GNN models, and inspect predictions.</p>
    <div class="hero-card">
        <div class="hero-eyebrow">OGBN-Arxiv overview</div>
        <h2>Research, connected.</h2>
        <p>A visual command center for 169K papers and the citation relationships that shape their subject classifications.</p>
        <span class="hero-pill">GCN vs GAT · 40 subject classes</span>
    </div>
    """,
    unsafe_allow_html=True,
)

data, train_idx, valid_idx, test_idx = load_data()

if data is None:
    st.error(
        "The processed OGBN-Arxiv dataset has not been generated yet. "
        f"Expected file: `{DATASET_PATH}`"
    )
    st.info(
        "Download `processed_data.pkl` using the link in `data/README.md` and place it "
        "in `data/processed/`, or run the data-preparation notebook to generate it."
    )
    st.stop()

in_channels = data.x.size(-1)
num_classes = int(data.y.max().item()) + 1

gcn_model, gat_model = load_models(in_channels, num_classes)

if gcn_model is None or gat_model is None:
    st.error(
        "Could not find one or both trained model checkpoints. "
        f"Expected `{GCN_CKPT_PATH}` and `{GAT_CKPT_PATH}`."
    )
    st.stop()

y_true = data.y.cpu().numpy()
gcn_probs, gat_probs = compute_predictions(gcn_model, gat_model, data)
gcn_pred = gcn_probs.argmax(axis=1)
gat_pred = gat_probs.argmax(axis=1)

page = st.sidebar.radio(
    "NAVIGATION",
    ["Overview", "Model Performance", "Node Explorer", "Embedding Map"],
)

train_idx_np = train_idx.cpu().numpy()
valid_idx_np = valid_idx.cpu().numpy()
test_idx_np = test_idx.cpu().numpy()


# ------------------------------------------------------------
# PAGE 1 — GRAPH STATISTICS
# ------------------------------------------------------------
if page == "Overview":
    st.header("Network overview")

    stats = compute_graph_stats(data)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nodes", f"{stats['num_nodes']:,}")
    c2.metric("Edges", f"{stats['num_edges']:,}")
    c3.metric("Density", f"{stats['density']:.2e}")
    c4.metric("Connected components", f"{stats['num_components']:,}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Avg. degree", f"{stats['avg_degree']:.2f}")
    c6.metric("Median degree", f"{stats['median_degree']:.1f}")
    c7.metric("Max degree", f"{stats['max_degree']:,}")
    c8.metric("Largest component", f"{stats['largest_component_frac']*100:.1f}% of nodes")

    st.subheader("Degree Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(stats["degree_array"], bins=100, color="#7562c8", edgecolor="#6753b8", linewidth=0.2)
    ax.set_yscale("log")
    ax.set_xlabel("Node degree (in + out)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Degree distribution across all nodes")
    st.pyplot(style_chart(fig, ax), use_container_width=True)

    st.subheader("Class Distribution")
    class_counts = pd.Series(y_true).value_counts().sort_index()
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.bar(class_counts.index.astype(str), class_counts.values, color="#f36f9d")
    ax2.set_xlabel("Class index")
    ax2.set_ylabel("Number of papers")
    ax2.set_title("Papers per subject category")
    plt.xticks(rotation=90, fontsize=6)
    st.pyplot(style_chart(fig2, ax2), use_container_width=True)

    st.subheader("Data Splits")
    split_df = pd.DataFrame(
        {
            "Split": ["Train", "Validation", "Test"],
            "Nodes": [len(train_idx_np), len(valid_idx_np), len(test_idx_np)],
        }
    )
    split_df["Nodes"] = split_df["Nodes"].map(lambda value: f"{value:,}")
    render_summary_table(split_df)


# ------------------------------------------------------------
# PAGE 2 — MODEL PERFORMANCE
# ------------------------------------------------------------
elif page == "Model Performance":
    st.header("Model Performance")

    split_choice = st.radio("Evaluate on", ["Validation set", "Test set"], horizontal=True)
    idx = valid_idx_np if split_choice == "Validation set" else test_idx_np

    gcn_metrics = metrics_for(y_true, gcn_pred, idx)
    gat_metrics = metrics_for(y_true, gat_pred, idx)

    metrics_df = pd.DataFrame({"GCN": gcn_metrics, "GAT": gat_metrics}).T
    metrics_df.index.name = "Model"
    metrics_display = metrics_df.reset_index()
    for column in metrics_display.columns[1:]:
        metrics_display[column] = metrics_display[column].map(lambda value: f"{value:.4f}")
    render_summary_table(metrics_display)

    fig, ax = plt.subplots(figsize=(7, 4))
    metric_names = list(gcn_metrics.keys())
    xpos = np.arange(len(metric_names))
    width = 0.35
    ax.bar(xpos - width / 2, [gcn_metrics[m] for m in metric_names], width, label="GCN", color="#7562c8")
    ax.bar(xpos + width / 2, [gat_metrics[m] for m in metric_names], width, label="GAT", color="#f36f9d")
    ax.set_xticks(xpos)
    ax.set_xticklabels(metric_names, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title(f"GCN vs GAT — {split_choice}")
    ax.legend()
    st.pyplot(style_chart(fig, ax), use_container_width=True)

    st.subheader("Strengths / Weaknesses (fill in after inspecting results)")
    st.text_area(
        "Notes for the technical report",
        placeholder=(
            "e.g. GAT slightly outperforms GCN on precision for underrepresented classes, "
            "likely because attention lets it down-weight noisy citation neighbors..."
        ),
        height=120,
    )


# ------------------------------------------------------------
# PAGE 3 — NODE CLASSIFICATION RESULTS
# ------------------------------------------------------------
elif page == "Node Explorer":
    st.header("Node explorer")

    split_for_lookup = st.selectbox("Pick from", ["Test set", "Validation set", "Any node ID"])
    if split_for_lookup == "Test set":
        options = test_idx_np
    elif split_for_lookup == "Validation set":
        options = valid_idx_np
    else:
        options = None

    if options is not None:
        node_id = st.selectbox("Node (paper) ID", options.tolist())
    else:
        node_id = st.number_input(
            "Node (paper) ID", min_value=0, max_value=data.num_nodes - 1, value=int(test_idx_np[0])
        )
        node_id = int(node_id)

    true_class = int(y_true[node_id])
    gcn_class = int(gcn_pred[node_id])
    gat_class = int(gat_pred[node_id])

    c1, c2, c3 = st.columns(3)
    c1.metric("True class", true_class)
    c2.metric("GCN prediction", gcn_class, delta="Correct" if gcn_class == true_class else "Incorrect")
    c3.metric("GAT prediction", gat_class, delta="Correct" if gat_class == true_class else "Incorrect")

    st.subheader("Class Probability Breakdown (Top 10)")
    col1, col2 = st.columns(2)

    for col, probs, title in [(col1, gcn_probs, "GCN"), (col2, gat_probs, "GAT")]:
        node_probs = probs[node_id]
        top10 = np.argsort(-node_probs)[:10]
        fig, ax = plt.subplots(figsize=(5, 4))
        colors = ["#f36f9d" if c == true_class else "#7562c8" for c in top10]
        ax.barh([str(c) for c in top10][::-1], node_probs[top10][::-1], color=colors[::-1])
        ax.set_xlabel("Predicted probability")
        ax.set_title(f"{title} — top 10 classes (pink = true class)")
        col.pyplot(style_chart(fig, ax), use_container_width=True)

    st.subheader("Citation Neighborhood")
    edge_index_np = data.edge_index.cpu().numpy()
    cites = edge_index_np[1][edge_index_np[0] == node_id]
    cited_by = edge_index_np[0][edge_index_np[1] == node_id]
    st.write(f"This paper cites **{len(cites)}** papers and is cited by **{len(cited_by)}** papers.")
    if len(cited_by) > 0:
        neighbor_classes = y_true[cited_by]
        same_class_frac = (neighbor_classes == true_class).mean()
        st.write(f"Of papers citing it, **{same_class_frac*100:.1f}%** share its true class.")


# ------------------------------------------------------------
# PAGE 4 — EMBEDDING VISUALIZATIONS
# ------------------------------------------------------------
elif page == "Embedding Map":
    st.header("Embedding map")

    sample_size = st.slider("Number of nodes to visualize", 500, 10000, 5000, step=500)
    gcn_emb, gat_emb, sample_idx = compute_embeddings(gcn_model, gat_model, data, sample_size=sample_size)
    sample_labels = y_true[sample_idx]

    method = st.radio("Projection method", ["PCA (fast)", "t-SNE (slower)"], horizontal=True)

    def plot_embedding(emb_2d, labels, title):
        fig, ax = plt.subplots(figsize=(6, 5))
        scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap="tab20", s=6, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        fig.colorbar(scatter, ax=ax, label="Class")
        return style_chart(fig, ax)

    col1, col2 = st.columns(2)

    if method == "PCA (fast)":
        pca_gcn = PCA(n_components=2).fit_transform(gcn_emb[sample_idx])
        pca_gat = PCA(n_components=2).fit_transform(gat_emb[sample_idx])
        col1.pyplot(plot_embedding(pca_gcn, sample_labels, "GCN embeddings — PCA"), use_container_width=True)
        col2.pyplot(plot_embedding(pca_gat, sample_labels, "GAT embeddings — PCA"), use_container_width=True)
    else:
        if st.button("Run t-SNE (can take a minute or two)"):
            with st.spinner("Computing t-SNE projections..."):
                tsne_gcn = TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(
                    gcn_emb[sample_idx]
                )
                tsne_gat = TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(
                    gat_emb[sample_idx]
                )
            col1.pyplot(plot_embedding(tsne_gcn, sample_labels, "GCN embeddings — t-SNE"), use_container_width=True)
            col2.pyplot(plot_embedding(tsne_gat, sample_labels, "GAT embeddings — t-SNE"), use_container_width=True)
        else:
            st.info("Click the button above to compute t-SNE — it's slower than PCA so it runs on demand.")
