#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/kemove/zlf/FedSGD2"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE_NAME="$("$PYTHON_BIN" -c 'import sys, torch; ok=torch.cuda.is_available(); print(torch.cuda.get_device_name(0) if ok else "NO CUDA"); sys.exit(0 if ok else 1)')"
if [[ "$CUDA_DEVICE_NAME" != "NVIDIA GeForce RTX 4090" ]]; then
  echo "[fatal] expected NVIDIA GeForce RTX 4090, got: $CUDA_DEVICE_NAME" >&2
  exit 1
fi
export CUDA_DEVICE_NAME

echo "[env] CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-}"
echo "[env] CUDA_DEVICE=$CUDA_DEVICE_NAME"
echo "[env] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "[mode] sequential seeds: 2 3 4"
echo "[mode] seed0/seed1 will not be run"

json_get() {
  local path="$1"
  local expr="$2"
  "$PYTHON_BIN" - "$path" "$expr" <<'PY'
import json
import sys
path, expr = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as f:
    obj = json.load(f)
cur = obj
for part in expr.split("."):
    cur = cur[part]
print(cur)
PY
}

latest_checkpoint() {
  local train_dir="$1"
  find "$train_dir/checkpoints" -maxdepth 1 -type f -name 'round_*.pth' 2>/dev/null | sort | tail -n 1
}

train_complete() {
  local seed="$1"
  local train_dir="$ROOT_DIR/results/table_iv_ours_main1_aligned_seed${seed}_r1000_fromscratch"
  [[ -s "$train_dir/result.json" ]] || return 1
  [[ -s "$train_dir/final_train_checkpoint.pth" ]] || return 1
  [[ "$(json_get "$train_dir/result.json" status)" == "ok" ]] || return 1
  [[ "$(json_get "$train_dir/result.json" train_rounds_completed)" == "1000" ]] || return 1
  "$PYTHON_BIN" - "$train_dir/result.json" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    r = json.load(f)
if r.get("seed") != int(sys.argv[1].split("seed")[1].split("_")[0]):
    raise SystemExit(1)
if r.get("ASR") is None or r.get("clean_acc_before_unlearning") is None:
    raise SystemExit(1)
PY
}

unlearn_complete() {
  local seed="$1"
  local unlearn_dir="$ROOT_DIR/results/table_iv_ours_revised_protocol_seed${seed}_confirm"
  [[ -s "$unlearn_dir/summary.json" ]] || return 1
  [[ -s "$unlearn_dir/metrics_by_round.csv" ]] || return 1
  [[ -s "$unlearn_dir/COMPLETED" ]] || return 1
  [[ -s "$unlearn_dir/unlearned_checkpoint.pth" ]] || return 1
  [[ "$(json_get "$unlearn_dir/summary.json" status)" == "ok" ]] || return 1
}

run_train_and_backdoor_if_needed() {
  local seed="$1"
  local train_dir="$ROOT_DIR/results/table_iv_ours_main1_aligned_seed${seed}_r1000_fromscratch"
  if train_complete "$seed"; then
    echo "[$(date '+%F %T')] seed${seed} training/backdoor complete; skipping"
    echo "[$(date '+%F %T')] seed${seed} checkpoint=$train_dir/final_train_checkpoint.pth"
    return 0
  fi

  local resume_args=()
  local resume_ckpt
  resume_ckpt="$(latest_checkpoint "$train_dir" || true)"
  if [[ -n "$resume_ckpt" && ! -s "$train_dir/final_train_checkpoint.pth" ]]; then
    resume_args=(--resume_checkpoint "$resume_ckpt")
    echo "[$(date '+%F %T')] seed${seed} resuming training/backdoor from $resume_ckpt"
  else
    echo "[$(date '+%F %T')] seed${seed} starting training/backdoor from scratch"
  fi

  "$PYTHON_BIN" -u table_iv_ours_main1_aligned.py \
    --seed "$seed" \
    --dataset cifar10 \
    --n_client 50 \
    --n_server1 5 \
    --batch_size 64 \
    --dirichlet_alpha 1.0 \
    --model_type cifar10cnn_v2 \
    --compression_mode adaptive \
    --train_rounds 1000 \
    --target_client 0 \
    --target_class 9 \
    --poison_ratio 0.8 \
    --trigger_size 3 \
    --trigger_location bottom-right \
    --unlearning_rounds 0 \
    --clean_acc_threshold 1.0 \
    --output_dir "$train_dir" \
    --save_checkpoint_every 100 \
    --eval_batch_size 256 \
    --device cuda \
    "${resume_args[@]}"

  train_complete "$seed"
  echo "[$(date '+%F %T')] seed${seed} training/backdoor completed"
}

run_unlearn_if_needed() {
  local seed="$1"
  local train_dir="$ROOT_DIR/results/table_iv_ours_main1_aligned_seed${seed}_r1000_fromscratch"
  local checkpoint="$train_dir/final_train_checkpoint.pth"
  local unlearn_dir="$ROOT_DIR/results/table_iv_ours_revised_protocol_seed${seed}_confirm"

  if unlearn_complete "$seed"; then
    echo "[$(date '+%F %T')] seed${seed} unlearning complete; skipping"
    return 0
  fi
  if [[ -e "$unlearn_dir" ]]; then
    shopt -s nullglob
    local existing=("$unlearn_dir"/*)
    shopt -u nullglob
    if (( ${#existing[@]} > 0 )); then
      echo "[fatal] seed${seed} has partial unlearning output: $unlearn_dir" >&2
      exit 1
    fi
  fi

  echo "[$(date '+%F %T')] seed${seed} starting revised unlearning"
  echo "[$(date '+%F %T')] seed${seed} checkpoint=$checkpoint"
  "$PYTHON_BIN" -u table_iv_ours_revised_protocol_confirm.py \
    --seed "$seed" \
    --dataset cifar10 \
    --model_type cifar10cnn_v2 \
    --n_client 50 \
    --n_server1 5 \
    --batch_size 64 \
    --dirichlet_alpha 1.0 \
    --target_client 0 \
    --target_class 9 \
    --poison_ratio 0.8 \
    --trigger_size 3 \
    --trigger_location bottom-right \
    --checkpoint "$checkpoint" \
    --momentum 0 \
    --grad_clip 5 \
    --lambda_retain 5 \
    --unlearn_lr 0.0009 \
    --unlearning_rounds 10 \
    --output_dir "$unlearn_dir" \
    --device cuda

  "$PYTHON_BIN" -u scripts/normalize_ours_unlearn_outputs.py \
    --seed "$seed" \
    --unlearn-dir "$unlearn_dir" \
    --checkpoint "$checkpoint" \
    --cuda-device "$CUDA_DEVICE_NAME" \
    --threshold 0.10

  unlearn_complete "$seed"
  echo "[$(date '+%F %T')] seed${seed} unlearning completed"
}

for seed in 2 3 4; do
  echo "[$(date '+%F %T')] ===== seed${seed} begin ====="
  run_train_and_backdoor_if_needed "$seed"
  run_unlearn_if_needed "$seed"
  echo "[$(date '+%F %T')] ===== seed${seed} completed ====="
done

echo "[$(date '+%F %T')] all requested seeds completed"
