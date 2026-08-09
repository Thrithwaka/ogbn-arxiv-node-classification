# OGBN-Arxiv Node Classification with Graph Neural Networks

Node classification on a large-scale citation network using Graph Neural
Networks, with model comparison, interpretability analysis, and an
interactive monitoring dashboard.

## Overview

This project builds and compares two GNN architectures to classify research
papers into subject categories using both their content and their citation
relationships. It uses the **OGBN-Arxiv** dataset — a citation network of
~169K papers and ~1.1M citation edges, each paper represented by a 128-dim
feature vector derived from its title and abstract.

**Problem:** Predict the subject category of a paper using its own features
and its position in the citation graph — a graph learning problem, not a
plain tabular classification problem.

**Approach:**
- Two GNN architectures implemented and benchmarked (GCN + GraphSAGE/GAT)
- Full graph analytics pass (degree distribution, density, connected
  components) before any modeling
- Model interpretability via embedding visualization and attention/feature
  importance analysis
- Results surfaced through a Streamlit dashboard for non-technical stakeholders

## Features

- 📊 **Graph analytics** — structural analysis of the citation network
- 🧠 **Two GNN architectures** — GCN and GraphSAGE/GAT, trained and compared
- 📈 **Evaluation suite** — accuracy, precision, recall, F1 across val/test
- 🔍 **Explainability** — embedding visualization (PCA/t-SNE), attention
  weights, feature/neighborhood importance
- 🖥️ **Interactive dashboard** — Streamlit app for graph stats, model
  metrics, and live predictions

## Tech Stack

| Layer | Tools |
|-------|-------|
| Modeling | PyTorch, PyTorch Geometric |
| Data | OGB (Open Graph Benchmark), NumPy, Pandas |
| Graph analysis | NetworkX |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |
| Experiment tracking | Weights & Biases |

## Project Structure

```
ogbn-arxiv-gnn/
├── data/
│   ├── raw/                  # downloaded OGBN-Arxiv (gitignored)
│   └── processed/            # processed splits & normalized features (gitignored)
├── notebooks/
│   ├── 01_tensor_fundamentals.ipynb
│   ├── 02_graph_analysis.ipynb
│   └── 03_data_preparation.ipynb
├── src/
│   ├── models/                # GCN, GraphSAGE/GAT implementations
│   ├── data/                  # loading & preprocessing pipeline
│   └── utils/                 # metrics, plotting, explainability helpers
├── dashboard/                  # Streamlit app
├── reports/
│   └── figures/                # exported plots
├── models_checkpoints/          # saved trained weights (gitignored)
├── requirements.txt
└── README.md
```

## Getting Started

```bash
git clone https://github.com/<org>/ogbn-arxiv-gnn.git
cd ogbn-arxiv-gnn

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Run the notebooks in `notebooks/` in order to reproduce data prep and
analysis, or run the dashboard directly:

```bash
streamlit run dashboard/app.py
```

## Roadmap

- [ ] Tensor operations foundation
- [ ] Graph structural analysis (degree distribution, density, components)
- [ ] Data pipeline (splits, normalization)
- [ ] GCN implementation
- [ ] Second architecture (GraphSAGE or GAT)
- [ ] Training & hyperparameter tuning
- [ ] Evaluation & model comparison
- [ ] Explainability analysis
- [ ] Dashboard
- [ ] Final write-up & presentation

## Team

| Name | Focus Area |
|------|------------|
| [Aadhil](https://github.com/AadhilAslam88) | Tensor operations & foundations |
| [Thilani](https://github.com/ThilaniDilmani)| Graph analysis & data pipeline |
| [Thrithwaka](https://github.com/Thrithwaka) | GNN model development |
| [Navodya](https://github.com/NavodyaRupasinghe2003) | Training, optimization & evaluation |
| [Kaveesha](https://github.com/Kaveesha23dil) | Explainability & dashboard |

See [`CONTRIBUTORS.md`](./CONTRIBUTORS.md) for detailed ownership and the
contribution workflow.

## Contributing

- Branch per feature: `feature/<short-name>` (e.g. `feature/gcn-model`)
- PRs into `main` require at least one review before merging
- Commits follow [Conventional Commits](https://www.conventionalcommits.org/)
- Log training runs in `reports/experiment_log.md` (or W&B)

## Dataset

[OGBN-Arxiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv), from the
Open Graph Benchmark. Downloaded automatically via the `ogb` package —
raw data is not committed to this repository.

## License

MIT
