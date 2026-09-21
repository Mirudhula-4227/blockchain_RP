# BlockFed-IDS

Day-one foundation for a lightweight intrusion-detection baseline and later federated-learning experiments.

## Layout

- `data/raw/`: official dataset downloads, ignored by Git
- `data/processed/`: leakage-safe NumPy arrays and metadata, ignored by Git
- `src/data.py`: loading, identifier/leakage removal, deduplication, stratified split, encoding, scaling
- `src/models.py`: small PyTorch MLP and metrics
- `src/train_baseline.py`: centralized baseline and shared results CSV
- `results/`: metrics, model weights, and committed experiment CSVs
- `configs/`: reproducible experiment settings
- `notebooks/`: exploration only

## Setup

Python 3.10+ is required. Create an environment and install pinned dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Docker prerequisite: run `docker run --rm hello-world` after installing Docker Desktop. Fabric is intentionally deferred.

## Dataset handoff

The primary dataset is **Edge-IIoTset**. Its verified labeled ML CSV is stored locally as `data/raw/edge_iiotset.csv` and is excluded from Git. The standard multiclass target is `Attack_type`; the binary target is `Attack_label` and is excluded from features. The current ToN-IoT network run is retained as a comparison experiment. PB-fdGAN is retained as a paper/method reference until its actual dataset identity is confirmed.

The fallback source is **ToN-IoT**, the official UNSW Industry 4.0/IIoT dataset. Its official project page is [UNSW ToN-IoT Datasets](https://research.unsw.edu.au/projects/toniot-datasets), which links the academic-use download archive.

The official UNSW page describes raw, processed, train/test, feature-description, and ground-truth directories. Prefer the processed or train/test CSV that contains the selected label column. If the downloaded archive has multiple files, choose one documented dataset slice and record its exact filename in the experiment notes before combining files.

To process the primary dataset:

```bash
python -m src.data data/raw/edge_iiotset.csv data/processed/edge_iiotset --label-column Attack_type --drop-column Attack_label --drop-column frame.time --drop-column ip.src_host --drop-column ip.dst_host --drop-column http.request.full_uri --drop-column tcp.options --drop-column tcp.payload --drop-column tcp.srcport
```

The pipeline logs row and class counts at load, leaky-column removal, deduplication, label creation, and transformed split stages in `data/processed/edge_iiotset/metadata.json`. It removes duplicates before splitting and fits imputers, one-hot encoders, and scalers on training data only.

### Edge-IIoTset source

Download the labeled Edge-IIoTset CSV from the public [Kaggle dataset mirror](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot) and place it at `data/raw/edge_iiotset.csv`. The archive was confirmed to contain `Attack_type` and `Attack_label`. Verify any future replacement file with `python -c "import pandas as pd; print(pd.read_csv('data/raw/edge_iiotset.csv', nrows=0).columns.tolist())"` before running:

```bash
python -m src.data data/raw/edge_iiotset.csv data/processed/edge_iiotset --label-column Attack_type --drop-column Attack_label --drop-column frame.time --drop-column ip.src_host --drop-column ip.dst_host --drop-column http.request.full_uri --drop-column tcp.options --drop-column tcp.payload --drop-column tcp.srcport
python -m src.train_baseline --config configs/edge_iiotset.yaml --dataset-name edge_iiotset
```

This produces `results/edge_iiotset_results.csv` using the same schema as the ToN-IoT comparison.

### Clean FedAvg

Run the requested non-IID matrix of 10 and 20 clients with Dirichlet alpha values 0.1, 0.5, and 1.0:

```bash
python -m src.fl --config configs/baseline.yaml --dataset-name edge_iiotset --output results/edge_iiotset_fedavg_results.csv
```

Each row records the client count, alpha, global round, evaluation metrics, communication bytes, and round time. The `clients` column is `NA` for centralized experiments.

### Day 3 security primitives

`src/attacks.py` provides reproducible label flipping, sign-flip/scaling updates, and feature-trigger backdoors. `src/defense.py` provides cosine-similarity and norm screening, exponential reputation updates, committee validation by validation loss, and clean aggregation baselines: FedAvg, Krum, coordinate-wise median, and trimmed mean. These functions operate on PyTorch state dictionaries so the same pseudocode can be reused by the FL loop.

## Centralized baseline

```bash
python -m src.train_baseline --config configs/baseline.yaml
```

Outputs include a dataset-specific results CSV, a JSON classification report, and the PyTorch state dict. Every experiment must preserve this CSV schema:

```text
dataset,model,method,clients,attack,malicious_fraction,alpha,seed,round,accuracy,macro_f1,precision,recall,fpr,params,model_kb,bytes_per_round,round_time_s
```

`NA` is used where a field does not apply, such as attack and round fields for centralized training.
