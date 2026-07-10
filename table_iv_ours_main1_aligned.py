import argparse
import csv
import json
import math
import os
import random
import shutil
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

SCRIPT_DIR = Path(__file__).resolve().parent
import sys

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model.client import client
from model.data import loader as FedDataLoader
from model.server import server
from model.server1 import server1
from model.Unlearning import UnlearningModule
from table_iv_backdoor_utils import (
    ExactPoisonedDataset,
    evaluate_accuracy,
    evaluate_asr,
    make_triggered_tensor_dataset,
)


ENCRYPTION_KEY = "fed_sgd_secret_key_2024"


class NullLogger:
    def info(self, message):
        print(message)

    def warning(self, message):
        print(message)

    def debug(self, message):
        pass

    def log_gradient_info(self, grad_norm):
        print(f"[Gradient Info]  Gradient L2 Norm: {grad_norm:.6f}")

    def log_server_aggregate(self, server_type, info):
        if server_type == "intermediate":
            print(
                "[Server1 {rank:>2}] Aggregated gradients from {active_clients} active clients "
                "({forgotten_clients} forgotten, total samples: {total_samples})".format(**info)
            )

    def log_unlearning(self, message):
        print(message)

    def log_attack(self, *args, **kwargs):
        pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def infer_in_dim(dataset: str) -> int:
    return 1 if dataset == "mnist" else 3


def client_assignments(n_client: int, n_server1: int) -> List[List[int]]:
    per_server = n_client // n_server1
    assignments = []
    for i in range(n_server1):
        start = i * per_server
        end = n_client if i == n_server1 - 1 else (i + 1) * per_server
        assignments.append(list(range(start, end)))
    return assignments


def find_server_for_client(assignments: List[List[int]], target_client: int) -> int:
    for server_rank, ranks in enumerate(assignments):
        if target_client in ranks:
            return server_rank
    raise ValueError(f"target_client={target_client} is not assigned to any server1")


def build_client_loaders(args, fed_data, target_client: int):
    loaders = []
    poisoned_dataset = None
    target_client_size = None
    poison_num = None
    for client_id in range(args.n_client):
        if client_id == target_client:
            base_subset = Subset(fed_data.train_dataset, fed_data.client_train_indices[client_id])
            poisoned_dataset = ExactPoisonedDataset(
                base_subset,
                target_class=args.target_class,
                poison_ratio=args.poison_ratio,
                seed=args.seed,
                trigger_size=args.trigger_size,
                trigger_location=args.trigger_location,
            )
            target_client_size = len(poisoned_dataset)
            poison_num = poisoned_dataset.poison_num
            generator = torch.Generator()
            generator.manual_seed(args.seed * 100000 + client_id)
            train_loader = DataLoader(
                poisoned_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                generator=generator,
            )
            test_loader = DataLoader(fed_data.test_dataset, batch_size=args.batch_size, shuffle=False)
        else:
            train_loader, test_loader = fed_data.get_loader(client_id, use_global_test=True)
        loaders.append((train_loader, test_loader))
    return loaders, poisoned_dataset, target_client_size, poison_num


def copy_checkpoint(cache_dir: Path, checkpoint_path: Path) -> None:
    src = cache_dir / "global_model_state.pkl"
    if src.exists():
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, checkpoint_path)


def load_checkpoint_to_cache(cache_dir: Path, checkpoint_path: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_path, cache_dir / "global_model_state.pkl")


def append_train_log(path: Path, row: Dict) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "round",
                "clean_acc",
                "clean_acc_percent",
                "checkpoint_path",
                "elapsed_seconds",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def ensure_unlearn_payload_metadata(cache_dir: Path, client_rank: int, compression_mode: str) -> None:
    path = cache_dir / f"grads_{client_rank}.pkl"
    payload = torch.load(path, map_location="cpu")
    enabled = compression_mode != "none"
    payload["compression_mode"] = compression_mode if enabled else "none"
    payload["encrypted"] = enabled
    payload["enable_compression_all"] = enabled
    payload["enable_compression"] = enabled
    payload["fixed_batch"] = False
    torch.save(payload, path)


def evaluate_clean_and_asr(model, fed_data, args, device) -> Tuple[float, float]:
    clean_loader = DataLoader(fed_data.test_dataset, batch_size=args.eval_batch_size, shuffle=False)
    triggered = make_triggered_tensor_dataset(
        fed_data.test_dataset,
        target_class=args.target_class,
        trigger_size=args.trigger_size,
        trigger_location=args.trigger_location,
        batch_size=args.eval_batch_size,
    )
    triggered_loader = DataLoader(triggered, batch_size=args.eval_batch_size, shuffle=False)
    clean_acc = evaluate_accuracy(model, clean_loader, device)
    asr = evaluate_asr(model, triggered_loader, args.target_class, device)
    return clean_acc, asr


def run(args) -> Dict:
    start_time = time.time()
    output_dir = Path(args.output_dir).resolve()
    workspace_dir = output_dir / "workspace"
    cache_dir = workspace_dir / "cache"
    checkpoints_dir = output_dir / "checkpoints"
    config_path = output_dir / "config.json"
    result_path = output_dir / "result.json"
    train_log_path = output_dir / "train_log.csv"

    output_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["target_class_source"] = "unconfirmed_assumption_from_reference_CWT_code"
    config["notes"] = (
        "Table IV Ours/adaptive runner aligned to FedSGD2/main_1.py modules. "
        "Original main_1.py is not modified. Original module cache paths are isolated "
        "by running inside output_dir/workspace."
    )
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    result = {
        "method": "Ours/adaptive",
        "seed": args.seed,
        "dataset": args.dataset,
        "n_client": args.n_client,
        "n_server1": args.n_server1,
        "batch_size": args.batch_size,
        "dirichlet_alpha": args.dirichlet_alpha,
        "model_type": args.model_type,
        "compression_mode": args.compression_mode,
        "train_rounds_completed": 0,
        "clean_acc_before_unlearning": None,
        "clean_acc_after_unlearning": None,
        "target_client": args.target_client,
        "target_client_size": None,
        "target_class": args.target_class,
        "target_class_source": config["target_class_source"],
        "poison_ratio": args.poison_ratio,
        "poison_num": None,
        "trigger_type": "white_pixel_patch",
        "trigger_size": args.trigger_size,
        "trigger_location": args.trigger_location,
        "unlearning_rounds": args.unlearning_rounds,
        "ASR": None,
        "ASR_percent": None,
        "checkpoint_path_before_unlearning": None,
        "checkpoint_path_after_unlearning": None,
        "elapsed_seconds": None,
        "status": "started",
        "error_if_any": None,
    }

    old_cwd = Path.cwd()
    try:
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

        if args.resume_checkpoint:
            load_checkpoint_to_cache(cache_dir, Path(args.resume_checkpoint).resolve())
            center_server.model.load_state_dict(torch.load(cache_dir / "global_model_state.pkl", map_location=device))

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

        client_loaders, _poisoned_dataset, target_size, poison_num = build_client_loaders(args, fed_data, args.target_client)
        result["target_client_size"] = target_size
        result["poison_num"] = poison_num

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

        for round_idx in range(1, args.train_rounds + 1):
            enable_compression = not (round_idx == 1 and not args.compress_first_round)
            enable_compression_all = enable_compression
            for c in clients:
                c.run(
                    enable_compression=enable_compression,
                    enable_compression_all=enable_compression_all,
                    use_fixed_batch=False,
                )
            for s1 in server1s:
                s1.aggregate(forgotten_clients=set(), epoch=round_idx, attack_rounds=[])
            center_server.aggregate(epoch=round_idx, attack_rounds=[])
            clean_acc = center_server.accuracy[-1] if center_server.accuracy else None
            result["train_rounds_completed"] = round_idx

            checkpoint_path = ""
            if args.save_checkpoint_every > 0 and (
                round_idx % args.save_checkpoint_every == 0
                or round_idx == args.train_rounds
                or (clean_acc is not None and clean_acc >= args.clean_acc_threshold)
            ):
                checkpoint_path = str(checkpoints_dir / f"round_{round_idx:04d}.pth")
                copy_checkpoint(cache_dir, Path(checkpoint_path))

            append_train_log(
                train_log_path,
                {
                    "round": round_idx,
                    "clean_acc": clean_acc,
                    "clean_acc_percent": clean_acc * 100.0 if clean_acc is not None else "",
                    "checkpoint_path": checkpoint_path,
                    "elapsed_seconds": time.time() - start_time,
                },
            )

            if clean_acc is not None and clean_acc >= args.clean_acc_threshold:
                break

        before_path = output_dir / "final_train_checkpoint.pth"
        copy_checkpoint(cache_dir, before_path)
        result["checkpoint_path_before_unlearning"] = str(before_path)
        result["clean_acc_before_unlearning"] = center_server.accuracy[-1] if center_server.accuracy else None

        if args.unlearning_rounds > 0:
            unlearning_module = UnlearningModule(
                args.n_client,
                args.n_server1,
                assignments,
                logger=logger,
                compression_mode=args.compression_mode,
            )
            unlearning_clients = {target_server: [args.target_client]}
            for _ in range(args.unlearning_rounds):
                clients[args.target_client].unlearn(enable_compression=args.compression_mode != "none")
                ensure_unlearn_payload_metadata(cache_dir, args.target_client, args.compression_mode)
                for server_rank, s1 in enumerate(server1s):
                    if server_rank == target_server:
                        s1.aggregate_unlearning_only(
                            unlearning_client_ranks=[args.target_client],
                            forgotten_clients=unlearning_module.get_forgotten_clients(),
                        )
                center_server.aggregate()

        after_path = output_dir / "unlearned_checkpoint.pth"
        copy_checkpoint(cache_dir, after_path)
        result["checkpoint_path_after_unlearning"] = str(after_path)
        clean_after, asr = evaluate_clean_and_asr(center_server.model, fed_data, args, device)
        result["clean_acc_after_unlearning"] = clean_after
        result["ASR"] = asr
        result["ASR_percent"] = asr * 100.0
        result["status"] = "ok"
    except Exception as exc:
        result["status"] = "error"
        result["error_if_any"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    finally:
        os.chdir(old_cwd)
        result["elapsed_seconds"] = time.time() - start_time
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        summary_path = output_dir / "ours_seed0_summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "method",
                    "seed",
                    "status",
                    "train_rounds_completed",
                    "clean_acc_before_unlearning",
                    "clean_acc_after_unlearning",
                    "ASR",
                    "ASR_percent",
                    "target_client_size",
                    "poison_num",
                    "result_path",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "method": result["method"],
                    "seed": result["seed"],
                    "status": result["status"],
                    "train_rounds_completed": result["train_rounds_completed"],
                    "clean_acc_before_unlearning": result["clean_acc_before_unlearning"],
                    "clean_acc_after_unlearning": result["clean_acc_after_unlearning"],
                    "ASR": result["ASR"],
                    "ASR_percent": result["ASR_percent"],
                    "target_client_size": result["target_client_size"],
                    "poison_num": result["poison_num"],
                    "result_path": str(result_path),
                }
            )

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Table IV Ours/adaptive runner aligned with FedSGD2/main_1.py.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100", "mnist"])
    parser.add_argument("--n_client", type=int, default=50)
    parser.add_argument("--n_server1", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dirichlet_alpha", type=float, default=1.0)
    parser.add_argument("--model_type", default="cifar10cnn_v2", choices=["lenet5", "cifar10cnn_v2", "resnet18"])
    parser.add_argument("--compression_mode", default="adaptive", choices=["adaptive", "fixed", "none"])
    parser.add_argument("--train_rounds", type=int, default=100)
    parser.add_argument("--target_client", type=int, default=0)
    parser.add_argument("--target_class", type=int, default=9)
    parser.add_argument("--poison_ratio", type=float, default=0.8)
    parser.add_argument("--trigger_size", type=int, default=3)
    parser.add_argument("--trigger_location", default="bottom-right")
    parser.add_argument("--unlearning_rounds", type=int, default=10)
    parser.add_argument("--clean_acc_threshold", type=float, default=0.60)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--save_checkpoint_every", type=int, default=10)
    parser.add_argument("--resume_checkpoint", default=None)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compress_first_round", action="store_true")
    parser.add_argument("--use_augmentation", action="store_true")
    return parser.parse_args()


def main():
    result = run(parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result.get("status") == "ok" else 1)


if __name__ == "__main__":
    main()
