import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model.client import client
from model.data import loader as FedDataLoader
from model.server import server
from model.server1 import server1
from model.Unlearning import UnlearningModule
from table_iv_backdoor_utils import ExactPoisonedDataset
from table_iv_ours_main1_aligned import (
    ENCRYPTION_KEY,
    NullLogger,
    build_client_loaders,
    client_assignments,
    ensure_unlearn_payload_metadata,
    evaluate_clean_and_asr,
    find_server_for_client,
    infer_in_dim,
    load_checkpoint_to_cache,
    set_seed,
)


def parse_sweep_rounds(text: str) -> List[int]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 0:
            raise ValueError("sweep rounds must be non-negative")
        values.append(value)
    if not values:
        raise ValueError("at least one sweep round is required")
    return values


def write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def copy_source_cache(source_workspace: Path, dest_workspace: Path, n_server1: int, n_client: int) -> Dict:
    source_cache = source_workspace / "cache"
    dest_cache = dest_workspace / "cache"
    if not source_cache.is_dir():
        raise FileNotFoundError(f"source cache not found: {source_cache}")

    missing = []
    for i in range(n_server1):
        if not (source_cache / f"grads_agg1_{i}.pkl").exists():
            missing.append(f"grads_agg1_{i}.pkl")
    for i in range(n_client):
        if not (source_cache / f"grads_{i}.pkl").exists():
            missing.append(f"grads_{i}.pkl")
    for name in ["global_model_state.pkl", "normalized_direction.pkl"]:
        if not (source_cache / name).exists():
            missing.append(name)
    if missing:
        raise FileNotFoundError(f"source cache is incomplete; missing: {missing[:20]}")

    if dest_cache.exists():
        shutil.rmtree(dest_cache)
    shutil.copytree(source_cache, dest_cache)

    return {
        "source_cache": str(source_cache),
        "dest_cache": str(dest_cache),
        "copied_files": len(list(dest_cache.iterdir())),
        "normalized_direction_copied": (dest_cache / "normalized_direction.pkl").exists(),
        "cache_limitation": (
            "The source workspace is copied as available from the completed 1000-round run. "
            "The original pre-unlearning normalized_direction is not separately snapshotted, "
            "so this diagnostic records and uses the available cache state rather than silently "
            "inventing a missing cache."
        ),
    }


def checkpoint_has_optimizer(checkpoint_path: Path) -> bool:
    obj = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(obj, dict):
        return False
    keys = [str(k).lower() for k in obj.keys()]
    return any("optimizer" in k for k in keys)


def load_model_state(checkpoint_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    obj = torch.load(checkpoint_path, map_location=device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj


def state_l2_delta(before: Dict[str, torch.Tensor], after: Dict[str, torch.Tensor]) -> float:
    total = 0.0
    with torch.no_grad():
        for key, value in before.items():
            if key not in after:
                continue
            delta = after[key].detach().cpu() - value.detach().cpu()
            total += float(delta.norm(2).item() ** 2)
    return math.sqrt(total)


def compute_unlearn_loss(model, loader, device: torch.device) -> float:
    criterion = nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    total = 0.0
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            total += float(criterion(model(data), target).item())
    return total


def decode_payload_grads(payload: Dict, center_server, model) -> Tuple[Dict[str, torch.Tensor], bool]:
    named_grads = payload["named_grads"]
    encrypted = bool(payload.get("encrypted", False))
    if not encrypted:
        return named_grads, False
    original_shapes = {k: v.shape for k, v in model.named_parameters()}
    decoded = center_server.compressor.decompress_with_decryption(
        named_grads,
        original_shapes,
        encrypted=True,
    )
    return decoded, True


def grad_stats_from_payload(cache_dir: Path, client_rank: int, center_server, model) -> Dict:
    payload_path = cache_dir / f"grads_{client_rank}.pkl"
    payload = torch.load(payload_path, map_location="cpu")
    grads, decrypted = decode_payload_grads(payload, center_server, model)
    total_sq = 0.0
    total_elems = 0
    nonzero_elems = 0
    for grad in grads.values():
        cpu_grad = grad.detach().cpu()
        total_sq += float(cpu_grad.norm(2).item() ** 2)
        total_elems += int(cpu_grad.numel())
        nonzero_elems += int((cpu_grad != 0).sum().item())
    retained = nonzero_elems / max(total_elems, 1)
    discard = 1.0 - retained
    return {
        "grad_l2_norm": math.sqrt(total_sq),
        "sparsity_or_discard_ratio": discard,
        "retained_ratio_if_available": retained,
        "payload_encrypted": bool(payload.get("encrypted", False)),
        "payload_decrypted_for_stats": decrypted,
        "payload_compression_mode": payload.get("compression_mode"),
        "payload_path": str(payload_path),
    }


def append_round_log(path: Path, row: Dict) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "round",
                "clean_acc",
                "ASR",
                "ASR_percent",
                "unlearn_loss",
                "grad_l2_norm",
                "update_l2_norm",
                "sparsity_or_discard_ratio",
                "retained_ratio_if_available",
                "elapsed_seconds",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_system(args, workspace_dir: Path):
    os.chdir(workspace_dir)
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger = NullLogger()

    fed_data = FedDataLoader(
        args.dataset,
        batch_size=args.batch_size,
        n_clients=args.n_client,
        alpha=args.dirichlet_alpha,
        seed=args.seed,
        use_augmentation=args.use_augmentation,
    )
    assignments = client_assignments(args.n_client, args.n_server1)
    target_server = find_server_for_client(assignments, args.target_client)

    full_train_loader = DataLoader(fed_data.train_dataset, batch_size=args.batch_size, shuffle=True)
    full_test_loader = DataLoader(fed_data.test_dataset, batch_size=args.batch_size, shuffle=False)
    center_server = server(
        size=args.n_server1,
        data_loader=(full_train_loader, full_test_loader),
        device=device,
        encryption_key=ENCRYPTION_KEY,
        n_class=fed_data.n_classes,
        logger=logger,
        enable_attack=False,
        attack_types=[],
        compression_mode=args.compression_mode,
        in_dim=infer_in_dim(args.dataset),
        model_type=args.model_type,
    )

    server1s = [
        server1(
            rank=i,
            client_ranks=assignments[i],
            logger=logger,
            enable_attack=False,
            attack_types=[],
            n_class=fed_data.n_classes,
            device=device,
            data_loader=fed_data,
        )
        for i in range(args.n_server1)
    ]

    client_loaders, poisoned_dataset, target_size, poison_num = build_client_loaders(
        args, fed_data, args.target_client
    )
    clients = []
    for i in range(args.n_client):
        c = client(
            rank=i,
            data_loader=client_loaders[i],
            device=device,
            encryption_key=ENCRYPTION_KEY,
            n_class=fed_data.n_classes,
            logger=logger,
            compression_mode=args.compression_mode,
            in_dim=infer_in_dim(args.dataset),
            model_type=args.model_type,
        )
        c.unlearn_mode = "unlearn"
        clients.append(c)

    return {
        "device": device,
        "logger": logger,
        "fed_data": fed_data,
        "assignments": assignments,
        "target_server": target_server,
        "center_server": center_server,
        "server1s": server1s,
        "clients": clients,
        "client_loaders": client_loaders,
        "poisoned_dataset": poisoned_dataset,
        "target_client_size": target_size,
        "poison_num": poison_num,
    }


def run_one_setting(args, sweep_round: int, optimizer_state_loaded: bool) -> Dict:
    start = time.time()
    output_dir = Path(args.output_dir).resolve()
    setting_dir = output_dir / f"rounds_{sweep_round}"
    workspace_dir = setting_dir / "workspace"
    cache_dir = workspace_dir / "cache"
    per_round_log = setting_dir / "per_round_log.csv"
    result_path = setting_dir / "result.json"
    config_path = setting_dir / "config.json"

    if setting_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing setting directory: {setting_dir}")
    workspace_dir.mkdir(parents=True, exist_ok=False)

    config = vars(args).copy()
    config["unlearning_rounds"] = sweep_round
    config["target_class_source"] = "unconfirmed_assumption_from_reference_CWT_code"
    config["notes"] = (
        "Offline unlearn-only diagnostic. No training is run. Each setting starts from "
        "the same 1000-round checkpoint and an isolated copy of the source workspace cache."
    )

    result = {
        "status": "started",
        "error_if_any": None,
        "method": "Ours/adaptive",
        "seed": args.seed,
        "dataset": args.dataset,
        "model_type": args.model_type,
        "compression_mode": args.compression_mode,
        "n_client": args.n_client,
        "n_server1": args.n_server1,
        "batch_size": args.batch_size,
        "dirichlet_alpha": args.dirichlet_alpha,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "source_workspace": str(Path(args.source_workspace).resolve()),
        "target_client": args.target_client,
        "target_server": None,
        "target_client_size": None,
        "target_class": args.target_class,
        "target_class_source": config["target_class_source"],
        "poison_ratio": args.poison_ratio,
        "poison_num": None,
        "poison_num_expected": None,
        "trigger_type": "white_pixel_patch",
        "trigger_size": args.trigger_size,
        "trigger_location": args.trigger_location,
        "unlearning_rounds": sweep_round,
        "clean_acc_before_unlearning": None,
        "clean_acc_after_unlearning": None,
        "ASR": None,
        "ASR_percent": None,
        "optimizer_state_loaded": optimizer_state_loaded,
        "optimizer_reset": not optimizer_state_loaded,
        "lr": None,
        "momentum": None,
        "elapsed_seconds": None,
        "per_round_log": str(per_round_log),
        "config_path": str(config_path),
        "result_path": str(result_path),
    }

    old_cwd = Path.cwd()
    try:
        cache_info = copy_source_cache(
            Path(args.source_workspace).resolve(),
            workspace_dir,
            args.n_server1,
            args.n_client,
        )
        config["cache_copy"] = cache_info
        result["cache_copy"] = cache_info

        system = build_system(args, workspace_dir)
        center_server = system["center_server"]
        device = system["device"]
        fed_data = system["fed_data"]
        server1s = system["server1s"]
        clients = system["clients"]
        target_server = system["target_server"]
        target_loader = system["client_loaders"][args.target_client][0]

        load_checkpoint_to_cache(cache_dir, Path(args.checkpoint).resolve())
        model_state = load_model_state(Path(args.checkpoint).resolve(), device)
        center_server.model.load_state_dict(model_state)

        lr = center_server.optimizer.param_groups[0].get("lr")
        momentum = center_server.optimizer.param_groups[0].get("momentum", 0.0)
        result["lr"] = lr
        result["momentum"] = momentum
        result["target_server"] = target_server
        result["target_client_size"] = system["target_client_size"]
        result["poison_num"] = system["poison_num"]
        result["poison_num_expected"] = int(math.floor(args.poison_ratio * system["target_client_size"]))
        config.update(
            {
                "target_server": target_server,
                "target_client_size": system["target_client_size"],
                "poison_num": system["poison_num"],
                "poison_num_expected": result["poison_num_expected"],
                "optimizer_state_loaded": optimizer_state_loaded,
                "optimizer_reset": not optimizer_state_loaded,
                "lr": lr,
                "momentum": momentum,
            }
        )
        write_json(config_path, config)

        clean_acc, asr = evaluate_clean_and_asr(center_server.model, fed_data, args, device)
        result["clean_acc_before_unlearning"] = clean_acc
        append_round_log(
            per_round_log,
            {
                "round": 0,
                "clean_acc": clean_acc,
                "ASR": asr,
                "ASR_percent": asr * 100.0,
                "unlearn_loss": "",
                "grad_l2_norm": "",
                "update_l2_norm": "",
                "sparsity_or_discard_ratio": "",
                "retained_ratio_if_available": "",
                "elapsed_seconds": time.time() - start,
            },
        )

        if sweep_round > 0:
            unlearning_module = UnlearningModule(
                args.n_client,
                args.n_server1,
                system["assignments"],
                logger=system["logger"],
                compression_mode=args.compression_mode,
            )
            for round_idx in range(1, sweep_round + 1):
                unlearn_loss = compute_unlearn_loss(center_server.model, target_loader, device)
                before_state = {k: v.detach().cpu().clone() for k, v in center_server.model.state_dict().items()}

                clients[args.target_client].unlearn(enable_compression=args.compression_mode != "none")
                ensure_unlearn_payload_metadata(cache_dir, args.target_client, args.compression_mode)
                grad_stats = grad_stats_from_payload(cache_dir, args.target_client, center_server, center_server.model)
                server1s[target_server].aggregate_unlearning_only(
                    unlearning_client_ranks=[args.target_client],
                    forgotten_clients=unlearning_module.get_forgotten_clients(),
                )
                center_server.aggregate()

                after_state = {k: v.detach().cpu().clone() for k, v in center_server.model.state_dict().items()}
                update_l2 = state_l2_delta(before_state, after_state)
                clean_acc, asr = evaluate_clean_and_asr(center_server.model, fed_data, args, device)
                append_round_log(
                    per_round_log,
                    {
                        "round": round_idx,
                        "clean_acc": clean_acc,
                        "ASR": asr,
                        "ASR_percent": asr * 100.0,
                        "unlearn_loss": unlearn_loss,
                        "grad_l2_norm": grad_stats["grad_l2_norm"],
                        "update_l2_norm": update_l2,
                        "sparsity_or_discard_ratio": grad_stats["sparsity_or_discard_ratio"],
                        "retained_ratio_if_available": grad_stats["retained_ratio_if_available"],
                        "elapsed_seconds": time.time() - start,
                    },
                )

        final_clean, final_asr = evaluate_clean_and_asr(center_server.model, fed_data, args, device)
        result["clean_acc_after_unlearning"] = final_clean
        result["ASR"] = final_asr
        result["ASR_percent"] = final_asr * 100.0
        torch.save(center_server.model.state_dict(), setting_dir / "unlearned_checkpoint.pth")
        result["checkpoint_path_after_unlearning"] = str(setting_dir / "unlearned_checkpoint.pth")
        result["status"] = "ok"
    except Exception as exc:
        result["status"] = "error"
        result["error_if_any"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        write_json(config_path, config)
    finally:
        os.chdir(old_cwd)
        result["elapsed_seconds"] = time.time() - start
        write_json(result_path, result)

    return result


def write_summary(output_dir: Path, rows: List[Dict]) -> None:
    csv_path = output_dir / "sweep_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "unlearning_rounds",
            "status",
            "clean_acc_before_unlearning",
            "clean_acc_after_unlearning",
            "ASR",
            "ASR_percent",
            "target_client_size",
            "poison_num",
            "elapsed_seconds",
            "result_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    write_json(output_dir / "sweep_summary.json", {"results": rows})


def parse_args():
    parser = argparse.ArgumentParser(description="Offline unlearn-only diagnostic for Table IV Ours/adaptive.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100", "mnist"])
    parser.add_argument("--n_client", type=int, default=50)
    parser.add_argument("--n_server1", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dirichlet_alpha", type=float, default=1.0)
    parser.add_argument("--model_type", default="cifar10cnn_v2", choices=["lenet5", "cifar10cnn_v2", "resnet18"])
    parser.add_argument("--compression_mode", default="adaptive", choices=["adaptive", "fixed", "none"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source_workspace", required=True)
    parser.add_argument("--target_client", type=int, default=0)
    parser.add_argument("--target_class", type=int, default=9)
    parser.add_argument("--poison_ratio", type=float, default=0.8)
    parser.add_argument("--trigger_size", type=int, default=3)
    parser.add_argument("--trigger_location", default="bottom-right")
    parser.add_argument("--sweep_rounds", default="0,1,2,3,4,5,10")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use_augmentation", action="store_true")
    args = parser.parse_args()
    args.sweep_rounds_list = parse_sweep_rounds(args.sweep_rounds)
    return args


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any((output_dir / f"rounds_{r}").exists() for r in args.sweep_rounds_list):
        raise SystemExit(f"Refusing to overwrite existing rounds_* directories under {output_dir}")

    checkpoint_path = Path(args.checkpoint).resolve()
    source_workspace = Path(args.source_workspace).resolve()
    if not checkpoint_path.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint_path}")
    if not source_workspace.exists():
        raise SystemExit(f"source_workspace not found: {source_workspace}")

    optimizer_state_loaded = checkpoint_has_optimizer(checkpoint_path)
    rows = []
    for sweep_round in args.sweep_rounds_list:
        result = run_one_setting(args, sweep_round, optimizer_state_loaded)
        rows.append(result)
        write_summary(output_dir, rows)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if result.get("status") != "ok":
            raise SystemExit(1)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
