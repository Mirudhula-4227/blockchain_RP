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

The base-paper dataset is **PB-fdGAN**. Download its labeled source file as `data/raw/pb_fdgan.csv` and set `label_column` plus any additional leaky target columns in `configs/baseline.yaml` after inspecting the file header. Do not commit raw data. The current ToN-IoT network run is retained as a fallback/comparison experiment, not as the base-paper result.

The fallback source is **ToN-IoT**, the official UNSW Industry 4.0/IIoT dataset. Its official project page is [UNSW ToN-IoT Datasets](https://research.unsw.edu.au/projects/toniot-datasets), which links the academic-use download archive.

The official UNSW page describes raw, processed, train/test, feature-description, and ground-truth directories. Prefer the processed or train/test CSV that contains the selected label column. If the downloaded archive has multiple files, choose one documented dataset slice and record its exact filename in the experiment notes before combining files.

Inspect and process it:

```bash
python -m src.data data/raw/pb_fdgan.csv data/processed/pb_fdgan --label-column label
```

The pipeline logs row and class counts at load, leaky-column removal, deduplication, label creation, and transformed split stages in `data/processed/pb_fdgan/metadata.json`. It removes duplicates before splitting and fits imputers, one-hot encoders, and scalers on training data only.

## Centralized baseline

```bash
python -m src.train_baseline --config configs/baseline.yaml
```

Outputs include a dataset-specific results CSV, a JSON classification report, and the PyTorch state dict. Every experiment must preserve this CSV schema:

```text
dataset,model,method,attack,malicious_fraction,alpha,seed,round,accuracy,macro_f1,precision,recall,fpr,params,model_kb,bytes_per_round,round_time_s
```

`NA` is used where a field does not apply, such as attack and round fields for centralized training.
