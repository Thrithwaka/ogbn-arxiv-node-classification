# data/

Raw and processed OGBN-Arxiv data are **not committed to this repository** —
the dataset and processed artifacts exceed GitHub's file size limits and
don't belong in git history (see project conventions in the main `README.md`).

## Processed data

`processed_data.pkl` — contains the preprocessed features, labels,
`edge_index`, and the official train/validation/test split indices produced
by `notebooks/03_data_preparation.ipynb`.

- **Download:** [Google Drive folder](https://drive.google.com/drive/folders/1nIOTVQYppk7ur2fUWjTmDwEaHgSNwWo5?usp=sharing)
- **Or regenerate locally:** run `notebooks/03_data_preparation.ipynb` end to
  end (downloads OGBN-Arxiv via the `ogb` package and rebuilds the file
  from scratch — takes a few minutes).

After downloading, place the file at: data/processed/processed_data.pkl

## Loading it in your own notebook/script

```python
import pickle

with open("data/processed/processed_data.pkl", "rb") as f:
    processed = pickle.load(f)

features   = processed["features"]
labels     = processed["labels"]
edge_index = processed["edge_index"]
train_idx  = processed["train_idx"]
valid_idx  = processed["valid_idx"]
test_idx   = processed["test_idx"]
```

## Raw data

The raw OGBN-Arxiv dataset (downloaded automatically by the `ogb` package)
is cached under `data/raw/` when you run the setup cell in any Task 02/03
notebook. It is gitignored and does not need to be shared manually —
everyone downloads it fresh from the Open Graph Benchmark the first time
they run the notebooks.

## Trained models

Trained model checkpoints (Task 04 onward) are not stored here either —
see `models_checkpoints/README.md` once models are available, or the
**Releases** section of this repository for published model weights.

## Folder structure

data/
├── raw/ # gitignored — downloaded automatically by ogb
├── processed/ # gitignored — download processed_data.pkl here (see above)
└── README.md # this file