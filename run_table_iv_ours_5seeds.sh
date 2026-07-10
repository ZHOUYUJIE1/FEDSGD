#!/usr/bin/env bash
set -euo pipefail

cd /home/kemove/zlf/FedSGD2

BASE_DIR="/home/kemove/zlf/FedSGD2/results/table_iv_ours_backdoor_full"
LOG_DIR="${BASE_DIR}/logs"
mkdir -p "$BASE_DIR" "$LOG_DIR"

echo "========== Table IV Ours/adaptive 5-seed run =========="
echo "Start time: $(date)"
echo "Base dir: $BASE_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
echo

for seed in 0 1 2 3 4; do
  echo "========== Running seed ${seed} =========="
  echo "Seed ${seed} start time: $(date)"

  OUT_DIR="${BASE_DIR}/seed_${seed}"
  mkdir -p "$OUT_DIR"

  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python3 /home/kemove/zlf/FedSGD2/table_iv_ours_backdoor_runner.py \
    --method adaptive \
    --seed "$seed" \
    --n_client 8 \
    --n_server1 2 \
    --batch_size 1 \
    --dirichlet_alpha 0.3 \
    --target_class 9 \
    --poison_ratio 0.8 \
    --trigger_size 3 \
    --trigger_location bottom-right \
    --unlearning_rounds 10 \
    --max_train_rounds 4 \
    --output_dir "$OUT_DIR" \
    --full_output_base "$BASE_DIR" \
    > "${LOG_DIR}/seed_${seed}.log" 2>&1

  echo "Seed ${seed} finished time: $(date)"

  if [ ! -f "${OUT_DIR}/dry_run_result.json" ] && [ ! -f "${OUT_DIR}/result.json" ]; then
    echo "ERROR: seed ${seed} result json not found in ${OUT_DIR}"
    exit 1
  fi
  echo
done

echo "========== Aggregating Ours ASR results =========="
python3 - <<'PY'
import json
import csv
import statistics
from pathlib import Path

base = Path("/home/kemove/zlf/FedSGD2/results/table_iv_ours_backdoor_full")
rows = []

for seed in range(5):
    seed_dir = base / f"seed_{seed}"
    candidates = [
        seed_dir / "dry_run_result.json",
        seed_dir / "result.json",
        seed_dir / "full_result.json",
    ]
    candidates += sorted(seed_dir.glob("*result*.json"))

    result_path = None
    for p in candidates:
        if p.exists():
            result_path = p
            break
    if result_path is None:
        raise FileNotFoundError(f"No result json found for seed {seed}: {seed_dir}")

    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    asr = data.get("ASR", data.get("asr", None))
    if asr is None:
        raise ValueError(f"ASR missing in {result_path}")
    asr = float(asr)
    asr_percent = asr * 100 if asr <= 1.0 else asr

    clean_acc = data.get("clean_accuracy_if_available", data.get("clean_acc", ""))
    if isinstance(clean_acc, (int, float)) and clean_acc <= 1.0:
        clean_acc = clean_acc * 100

    rows.append({
        "method": data.get("method", "Ours/adaptive"),
        "seed": seed,
        "asr": asr,
        "asr_percent": asr_percent,
        "clean_acc_percent": clean_acc,
        "target_class": data.get("target_class", ""),
        "target_client": data.get("target_client", ""),
        "target_client_size": data.get("target_client_size", ""),
        "poison_num": data.get("poison_num", ""),
        "config_path": str(seed_dir / "config.json"),
        "result_path": str(result_path),
    })

raw_path = base / "ours_asr_raw.csv"
summary_path = base / "ours_asr_summary.csv"

with open(raw_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

asrs = [r["asr_percent"] for r in rows]
mean_asr = statistics.mean(asrs)
std_asr = statistics.stdev(asrs) if len(asrs) > 1 else 0.0

with open(summary_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "method", "mean_asr_percent", "std_asr_percent", "num_seeds"
    ])
    writer.writeheader()
    writer.writerow({
        "method": "Ours/adaptive",
        "mean_asr_percent": mean_asr,
        "std_asr_percent": std_asr,
        "num_seeds": len(asrs),
    })

print("raw:", raw_path)
print("summary:", summary_path)
print(f"Ours ASR mean ± std: {mean_asr:.2f} ± {std_asr:.2f}")
PY

echo "End time: $(date)"
echo "========== Done =========="
