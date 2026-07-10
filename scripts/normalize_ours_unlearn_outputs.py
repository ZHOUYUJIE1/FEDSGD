#!/usr/bin/env python3
import argparse
import csv
import json
import os
import time
from pathlib import Path


METRIC_FIELDS = [
    "seed",
    "round",
    "clean_accuracy",
    "ASR",
    "learning_rate",
    "forget_ce",
    "retain_ce",
    "total_loss",
    "gradient_norm",
    "update_norm",
    "clean_drop_pp",
    "ASR_drop_pp",
    "threshold_reached",
]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize Table IV Ours unlearning outputs.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--unlearn-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cuda-device", required=True)
    parser.add_argument("--threshold", type=float, default=0.10)
    args = parser.parse_args()

    unlearn_dir = Path(args.unlearn_dir).resolve()
    config_path = unlearn_dir / "config.json"
    result_path = unlearn_dir / "result.json"
    per_round_path = unlearn_dir / "per_round_log.csv"
    checkpoint_path = unlearn_dir / "unlearned_checkpoint.pth"

    for path in [config_path, result_path, per_round_path, checkpoint_path]:
        if not path.exists():
            raise FileNotFoundError(f"missing required output: {path}")

    config = read_json(config_path)
    result = read_json(result_path)
    if result.get("status") != "ok":
        raise RuntimeError(f"unlearning did not finish cleanly: {result.get('status')} {result.get('error_if_any')}")

    metrics_path = unlearn_dir / "metrics_by_round.csv"
    first_threshold_round = None
    with per_round_path.open("r", newline="", encoding="utf-8") as src, metrics_path.open(
        "w", newline="", encoding="utf-8"
    ) as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for row in reader:
            asr = float(row["ASR"])
            round_idx = int(row["round"])
            threshold_reached = asr <= args.threshold
            if threshold_reached and first_threshold_round is None:
                first_threshold_round = round_idx
            writer.writerow(
                {
                    "seed": args.seed,
                    "round": round_idx,
                    "clean_accuracy": row["clean_acc"],
                    "ASR": row["ASR"],
                    "learning_rate": row["learning_rate"],
                    "forget_ce": row["forget_ce"],
                    "retain_ce": row["retain_ce"],
                    "total_loss": row["total_loss"],
                    "gradient_norm": row["grad_l2_norm"],
                    "update_norm": row["update_l2_norm"],
                    "clean_drop_pp": row["clean_drop_pp"],
                    "ASR_drop_pp": row["ASR_drop_pp"],
                    "threshold_reached": str(threshold_reached).lower(),
                }
            )

    summary = {
        "seed": args.seed,
        "status": result["status"],
        "checkpoint_source": str(Path(args.checkpoint).resolve()),
        "clean_before": result["clean_acc_before"],
        "ASR_before": result["ASR_before"],
        "clean_after": result["clean_acc_after"],
        "ASR_after": result["ASR_after"],
        "clean_drop_pp": result["clean_drop_pp"],
        "ASR_drop_pp": result["ASR_drop_pp"],
        "first_round_ASR_le_10_percent": result.get("first_asr_le_10_round", first_threshold_round),
        "target_client_size": result["target_client_size"],
        "poisoned_sample_count": result["poison_num"],
        "unlearning_params": {
            "unlearn_lr": config["unlearn_lr"],
            "grad_clip": config["grad_clip"],
            "lambda_retain": config["lambda_retain"],
            "momentum": config["momentum"],
            "unlearning_rounds": config["unlearning_rounds"],
            "poison_ratio": config["poison_ratio"],
            "target_client": config["target_client"],
            "target_class": config["target_class"],
            "trigger_size": config["trigger_size"],
            "trigger_location": config["trigger_location"],
            "ASR_completion_threshold": args.threshold,
        },
        "CUDA_device": args.cuda_device,
        "total_runtime_seconds": result["elapsed_seconds"],
        "normalized_at_unix": time.time(),
    }
    (unlearn_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (unlearn_dir / "COMPLETED").write_text(
        f"seed={args.seed}\nstatus=ok\ncheckpoint={args.checkpoint}\nCUDA_device={args.cuda_device}\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
