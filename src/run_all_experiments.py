"""Full multi-seed experiment runner across datasets, attacks, defenses, and adaptive threats."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import yaml

from src.fl import run_fedavg


def main() -> None:
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    datasets = [
        {"name": "edge_iiotset", "config": "configs/baseline.yaml"},
        {"name": "ton_iot_network", "config": "configs/baseline.yaml", "processed": "data/processed/primary"},
    ]
    
    seeds = [42, 43, 44]
    client_counts = [10]
    alphas = [0.5]
    
    attack_defense_pairs = [
        ("none", "fedavg"),
        ("label_flip", "fedavg"),
        ("label_flip", "similarity_norm"),
        ("label_flip", "reputation"),
        ("label_flip", "committee"),
        ("label_flip", "median"),
        ("label_flip", "trimmed_mean"),
        ("label_flip", "krum"),
        ("sign_flip", "fedavg"),
        ("sign_flip", "similarity_norm"),
        ("sign_flip", "reputation"),
        ("sign_flip", "committee"),
        ("sign_flip", "median"),
        ("sign_flip", "trimmed_mean"),
        ("sign_flip", "krum"),
        ("scaling", "fedavg"),
        ("scaling", "similarity_norm"),
        ("scaling", "reputation"),
        ("scaling", "committee"),
        ("scaling", "median"),
        ("scaling", "trimmed_mean"),
        ("scaling", "krum"),
        ("backdoor", "fedavg"),
        ("backdoor", "similarity_norm"),
        ("backdoor", "reputation"),
        ("backdoor", "committee"),
        ("backdoor", "median"),
        ("backdoor", "trimmed_mean"),
        ("backdoor", "krum"),
        ("adaptive", "fedavg"),
        ("adaptive", "similarity_norm"),
        ("adaptive", "reputation"),
        ("adaptive", "committee"),
    ]

    print("==> Starting Full Experimental Matrix Run...")
    for ds in datasets:
        ds_name = ds["name"]
        cfg_path = ds["config"]
        config = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
        
        proc_path = Path(ds.get("processed", config["processed_dir"]))
        if not proc_path.exists():
            print(f"Skipping {ds_name} - processed path {proc_path} does not exist")
            continue
            
        x_train = np.load(proc_path / "X_train.npy"); y_train = np.load(proc_path / "y_train.npy")
        x_val = np.load(proc_path / "X_val.npy"); y_val = np.load(proc_path / "y_val.npy")
        x_test = np.load(proc_path / "X_test.npy"); y_test = np.load(proc_path / "y_test.npy")
        metadata = json.loads((proc_path / "metadata.json").read_text(encoding="utf-8"))
        
        for attack, defense in attack_defense_pairs:
            out_file = results_dir / f"{ds_name}_{attack}_{defense}.csv"
            print(f"--> Running {ds_name} | Attack: {attack} | Defense: {defense} across seeds {seeds}")
            if out_file.exists():
                out_file.unlink()
            for s in seeds:
                for c in client_counts:
                    for a in alphas:
                        run_fedavg(
                            x_train, y_train, x_val, y_val, x_test, y_test,
                            x_train.shape[1], len(metadata["class_mapping"]),
                            config, ds_name, c, a, out_file,
                            attack=attack, defense=defense, ledger_type="simulated", custom_seed=s
                        )

    print("\n✅ Full Experimental Matrix Run Completed Successfully!")


if __name__ == "__main__":
    main()
