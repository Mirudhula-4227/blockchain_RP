"""Verification helper for Person B to extract and cross-reference all numbers from results files."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def summarize_csv(file_path: Path) -> dict[str, float]:
    if not file_path.exists():
        return {}
    with file_path.open("r", encoding="utf-8") as h:
        reader = list(csv.DictReader(h))
    if not reader:
        return {}
    # Take final round metrics
    last = reader[-1]
    return {
        "accuracy": float(last.get("accuracy", 0)),
        "macro_f1": float(last.get("macro_f1", 0)),
        "precision": float(last.get("precision", 0)),
        "recall": float(last.get("recall", 0)),
        "fpr": float(last.get("fpr", 0)),
    }


def main() -> None:
    results_dir = Path("results")
    print("==========================================================================================")
    print("                    BLOCKFED-IDS EXPERIMENTAL METRICS VERIFICATION REPORT                 ")
    print("==========================================================================================")
    
    # 1. Centralized Baselines
    print("\n--- 1. Centralized Model Baselines ---")
    for ds in ["edge_iiotset", "ton_iot_network"]:
        metrics_fn = results_dir / f"{ds}_centralized_metrics.json"
        if metrics_fn.exists():
            data = json.loads(metrics_fn.read_text(encoding="utf-8"))
            print(f"Dataset: {ds:<18} | Accuracy: {data['accuracy']:.4f} | Macro-F1: {data['macro_f1']:.4f} | FPR: {data['fpr']:.4f}")

    # 2. Attack vs Defense Matrix (Edge-IIoTset)
    print("\n--- 2. Attack vs Defense Matrix (Edge-IIoTset Final Round) ---")
    print(f"{'Attack':<15} | {'Defense':<18} | {'Accuracy':<10} | {'Macro-F1':<10} | {'FPR':<10}")
    print("-" * 75)
    
    for fn in sorted(results_dir.glob("edge_iiotset_*.csv")):
        if "fedavg_results" in fn.name or fn.name == "edge_iiotset_results.csv":
            continue
        parts = fn.stem.replace("edge_iiotset_", "").split("_")
        metrics = summarize_csv(fn)
        if metrics:
            print(f"{fn.stem:<35} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['macro_f1']:.4f} | FPR: {metrics['fpr']:.4f}")

    # 3. Fabric Performance Metrics
    print("\n--- 3. Hyperledger Fabric Network & Edge Telemetry ---")
    fab_fn = results_dir / "fabric_performance_benchmark.json"
    if fab_fn.exists():
        data = json.loads(fab_fn.read_text(encoding="utf-8"))
        lat = data.get("latency_ms", {})
        print(f"SubmitUpdate Latency (mean): {lat.get('SubmitUpdate', {}).get('mean_ms', 0):.2f} ms")
        print(f"Status Query Latency (mean): {lat.get('Status', {}).get('mean_ms', 0):.2f} ms")
        print(f"Throughput (TPS):           {data.get('throughput_tps', {}).get('tps', 0)} transactions/sec")
        print(f"Bandwidth Reduction Factor: {data.get('storage', {}).get('bandwidth_reduction_factor', 0)}x")
        print(f"Edge Ed25519 Sign Time:     {data.get('raspberry_pi_edge_simulation', {}).get('ed25519_sign_time_ms', 0)} ms")

    print("==========================================================================================\n")


if __name__ == "__main__":
    main()
