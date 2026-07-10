import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


LR_VALUES = [0.00090, 0.000925, 0.00095, 0.000975, 0.00100]
LAMBDA_RETAIN_VALUES = [5.0, 6.0, 7.0]
TARGET_ASR = 0.0630
CURRENT_FRESH = {
    "unlearn_lr": 0.0009,
    "grad_clip": 5.0,
    "lambda_retain": 5.0,
    "clean_acc_after": 0.6010,
    "ASR_after": 0.0857,
}


SUMMARY_FIELDS = [
    "config_id",
    "status",
    "unlearn_lr",
    "grad_clip",
    "lambda_retain",
    "clean_acc_before",
    "ASR_before",
    "clean_acc_after",
    "ASR_after",
    "ASR_percent_after",
    "clean_drop_pp",
    "ASR_drop_pp",
    "elapsed_seconds",
    "output_dir",
    "error_if_any",
]


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def config_id(lr: float, lam: float) -> str:
    lr_part = f"{lr:.6f}".replace(".", "p")
    lam_part = f"{lam:g}".replace(".", "p")
    return f"lr{lr_part}_clip5_lam{lam_part}"


def run_one(args, lr: float, lam: float) -> dict:
    cid = config_id(lr, lam)
    out_dir = args.output_dir / cid
    cmd = [
        sys.executable,
        str(args.confirm_script),
        "--seed",
        "0",
        "--dataset",
        "cifar10",
        "--model_type",
        "cifar10cnn_v2",
        "--n_client",
        "50",
        "--n_server1",
        "5",
        "--batch_size",
        "64",
        "--dirichlet_alpha",
        "1.0",
        "--target_client",
        "0",
        "--target_class",
        "9",
        "--poison_ratio",
        "0.8",
        "--trigger_size",
        "3",
        "--trigger_location",
        "bottom-right",
        "--checkpoint",
        str(args.checkpoint),
        "--momentum",
        "0",
        "--grad_clip",
        "5",
        "--lambda_retain",
        f"{lam:g}",
        "--unlearn_lr",
        f"{lr:.6f}",
        "--unlearning_rounds",
        "10",
        "--output_dir",
        str(out_dir),
        "--device",
        args.device,
    ]
    print(f"[local-calib] start {cid}", flush=True)
    completed = subprocess.run(
        cmd,
        cwd=args.work_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (out_dir / "run_stdout.log").write_text(completed.stdout, encoding="utf-8")

    result_path = out_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        result = {
            "status": "error",
            "error_if_any": f"missing result.json; returncode={completed.returncode}",
            "unlearn_lr": lr,
            "grad_clip": 5.0,
            "lambda_retain": lam,
        }

    row = {field: result.get(field) for field in SUMMARY_FIELDS}
    row.update(
        {
            "config_id": cid,
            "unlearn_lr": lr,
            "grad_clip": 5.0,
            "lambda_retain": lam,
            "output_dir": str(out_dir),
        }
    )
    print(
        "[local-calib] done "
        f"{cid} status={row['status']} "
        f"clean={row['clean_acc_after']} asr={row['ASR_after']}",
        flush=True,
    )
    if args.stop_on_error and row["status"] != "ok":
        raise RuntimeError(f"{cid} failed; see {out_dir / 'run_stdout.log'}")
    return row


def choose(rows: list[dict]) -> dict:
    ok_rows = [r for r in rows if r.get("status") == "ok"]

    def clean_at_least(threshold: float) -> list[dict]:
        return [r for r in ok_rows if float(r["clean_acc_after"]) >= threshold]

    def closest_to_target(candidates: list[dict]):
        if not candidates:
            return None
        return min(candidates, key=lambda r: abs(float(r["ASR_after"]) - TARGET_ASR))

    clean60 = clean_at_least(0.60)
    clean595 = clean_at_least(0.595)
    clean60_in_band = [
        r for r in clean60 if 0.06 <= float(r["ASR_after"]) <= 0.08
    ]
    clean60_lowest = min(clean60, key=lambda r: float(r["ASR_after"])) if clean60 else None
    clean60_closest = closest_to_target(clean60)
    closest595 = closest_to_target(clean595)
    band_choice = closest_to_target(clean60_in_band)

    recommended = band_choice or clean60_lowest
    if recommended is None:
        recommended = CURRENT_FRESH.copy()
        recommended["config_id"] = "current_fresh_confirmation"
        recommended["status"] = "fallback"

    improves_current = (
        recommended.get("status") == "ok"
        and float(recommended["clean_acc_after"]) >= CURRENT_FRESH["clean_acc_after"]
        and float(recommended["ASR_after"]) < CURRENT_FRESH["ASR_after"]
    )
    simple_stable_replacement = (
        recommended.get("status") == "ok"
        and float(recommended["clean_acc_after"]) >= 0.60
        and 0.06 <= float(recommended["ASR_after"]) <= 0.08
        and float(recommended["clean_acc_after"]) >= CURRENT_FRESH["clean_acc_after"] - 0.002
    )

    return {
        "target_ASR": TARGET_ASR,
        "current_fresh_confirmation": CURRENT_FRESH,
        "completed_configs": len(ok_rows),
        "total_configs": len(rows),
        "closest_clean_ge_60": clean60_closest,
        "closest_clean_ge_59p5": closest595,
        "clean_ge_60_asr_6_to_8": clean60_in_band,
        "clean_ge_60_lowest_asr": clean60_lowest,
        "recommended_candidate": recommended,
        "improves_current_fresh_confirmation": improves_current,
        "recommend_replace_revised_protocol": simple_stable_replacement,
        "recommend_enter_seed_1_or_5_seeds": False,
        "selection_note": (
            "Prefer clean_acc_after >= 60%, ASR in 6%-8%, and no meaningful clean drop. "
            "If this is not met, keep the current fresh confirmation protocol."
        ),
    }


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in SUMMARY_FIELDS})


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Fresh-run local calibration for the revised Table IV Ours protocol."
    )
    parser.add_argument("--work_dir", type=Path, default=root)
    parser.add_argument(
        "--confirm_script",
        type=Path,
        default=root / "table_iv_ours_revised_protocol_confirm.py",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root
        / "results/table_iv_ours_main1_aligned_seed0_r1000_fromscratch/final_train_checkpoint.pth",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=root / "results/table_iv_ours_revised_protocol_local_calib_seed0",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stop_on_error", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    start = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for lr in LR_VALUES:
        for lam in LAMBDA_RETAIN_VALUES:
            rows.append(run_one(args, lr, lam))
            write_summary_csv(args.output_dir / "local_calib_summary.csv", rows)
            write_json(args.output_dir / "local_calib_summary.json", choose(rows))

    summary = choose(rows)
    summary["elapsed_seconds"] = time.time() - start
    write_summary_csv(args.output_dir / "local_calib_summary.csv", rows)
    write_json(args.output_dir / "local_calib_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
