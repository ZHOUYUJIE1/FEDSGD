import argparse
import csv
import json
import math
import random
import time
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import ConcatDataset, DataLoader, Subset

from model.data import loader as FedDataLoader
from model.lenet import cifar10cnn_v2, lenet5, resnet18
from table_iv_backdoor_utils import (
    ExactPoisonedDataset,
    evaluate_accuracy,
    evaluate_asr,
    make_triggered_tensor_dataset,
)


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


def build_model(model_type: str, dataset: str, n_class: int):
    in_dim = infer_in_dim(dataset)
    if model_type == "resnet18":
        return resnet18(n_class=n_class, in_dim=in_dim)
    if model_type == "cifar10cnn_v2":
        return cifar10cnn_v2(n_class=n_class, in_dim=in_dim)
    return lenet5(n_class=n_class, in_dim=in_dim)


def load_state_dict(path: Path, device: torch.device):
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj


def write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def append_csv(path: Path, fieldnames: List[str], row: Dict) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def make_eval_loaders(fed_data, args):
    clean_loader = DataLoader(fed_data.test_dataset, batch_size=args.eval_batch_size, shuffle=False)
    triggered = make_triggered_tensor_dataset(
        fed_data.test_dataset,
        target_class=args.target_class,
        trigger_size=args.trigger_size,
        trigger_location=args.trigger_location,
        batch_size=args.eval_batch_size,
    )
    trigger_loader = DataLoader(triggered, batch_size=args.eval_batch_size, shuffle=False)
    return clean_loader, trigger_loader


def evaluate_pair(model, clean_loader, trigger_loader, target_class: int, device: torch.device):
    clean_acc = evaluate_accuracy(model, clean_loader, device)
    asr = evaluate_asr(model, trigger_loader, target_class, device)
    return clean_acc, asr


def grad_l2_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        total += float(param.grad.detach().norm(2).item() ** 2)
    return math.sqrt(total)


def state_l2_delta(before: Dict[str, torch.Tensor], after: Dict[str, torch.Tensor]) -> float:
    total = 0.0
    for key, value in before.items():
        if key not in after:
            continue
        delta = after[key].detach().cpu() - value.detach().cpu()
        total += float(delta.norm(2).item() ** 2)
    return math.sqrt(total)


def model_distance_loss(model, reference_state: Dict[str, torch.Tensor], device: torch.device):
    total = None
    for name, param in model.named_parameters():
        ref = reference_state[name].to(device)
        term = torch.sum((param - ref) ** 2)
        total = term if total is None else total + term
    return total if total is not None else torch.tensor(0.0, device=device)


def build_retain_loader(args, fed_data, target_client: int):
    datasets = []
    for client_id, indices in enumerate(fed_data.client_train_indices):
        if client_id == target_client:
            continue
        datasets.append(Subset(fed_data.train_dataset, indices))
    retain_dataset = ConcatDataset(datasets)
    generator = torch.Generator()
    generator.manual_seed(args.seed + 424242)
    return DataLoader(
        retain_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )


def next_retain_batch(retain_iter, retain_loader):
    try:
        return next(retain_iter), retain_iter
    except StopIteration:
        retain_iter = iter(retain_loader)
        return next(retain_iter), retain_iter


def build_grid():
    grid = []
    grid.append(
        {
            "name": "baseline_current_lr001_m09",
            "unlearn_lr": 0.01,
            "momentum": 0.9,
            "grad_clip": None,
            "proximal_mu": 0.0,
            "retain_reg": False,
            "lambda_retain": 0.0,
            "rounds": 10,
            "objective": "-CE(forget_poisoned)",
            "algorithm_modification": False,
        }
    )
    for lr in [0.003, 0.001, 0.0005, 0.0001]:
        grid.append(
            {
                "name": f"low_lr_{lr:g}_m0",
                "unlearn_lr": lr,
                "momentum": 0.0,
                "grad_clip": None,
                "proximal_mu": 0.0,
                "retain_reg": False,
                "lambda_retain": 0.0,
                "rounds": 10,
                "objective": "-CE(forget_poisoned)",
                "algorithm_modification": True,
            }
        )
    for lr in [0.001, 0.0005]:
        for clip in [10.0, 5.0, 1.0]:
            grid.append(
                {
                    "name": f"clip_lr{lr:g}_c{clip:g}",
                    "unlearn_lr": lr,
                    "momentum": 0.0,
                    "grad_clip": clip,
                    "proximal_mu": 0.0,
                    "retain_reg": False,
                    "lambda_retain": 0.0,
                    "rounds": 10,
                    "objective": "-CE(forget_poisoned) with grad clipping",
                    "algorithm_modification": True,
                }
            )
    for lr in [0.001, 0.0005]:
        for mu in [0.001, 0.01, 0.1]:
            grid.append(
                {
                    "name": f"prox_lr{lr:g}_mu{mu:g}",
                    "unlearn_lr": lr,
                    "momentum": 0.0,
                    "grad_clip": 5.0,
                    "proximal_mu": mu,
                    "retain_reg": False,
                    "lambda_retain": 0.0,
                    "rounds": 10,
                    "objective": "-CE(forget_poisoned) + 0.5 * proximal_mu * ||theta-theta0||^2",
                    "algorithm_modification": True,
                }
            )
    for lam in [0.1, 0.5, 1.0]:
        grid.append(
            {
                "name": f"retain_lr0005_lam{lam:g}",
                "unlearn_lr": 0.0005,
                "momentum": 0.0,
                "grad_clip": 5.0,
                "proximal_mu": 0.0,
                "retain_reg": True,
                "lambda_retain": lam,
                "rounds": 10,
                "objective": "-CE(forget_poisoned) + lambda_retain * CE(retain_clean)",
                "algorithm_modification": True,
            }
        )
    return grid


def run_config(args, cfg: Dict, common) -> Dict:
    start = time.time()
    safe_name = cfg["name"].replace(".", "p")
    config_dir = Path(args.output_dir).resolve() / safe_name
    if config_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing directory: {config_dir}")
    config_dir.mkdir(parents=True, exist_ok=False)

    config_path = config_dir / "config.json"
    result_path = config_dir / "result.json"
    log_path = config_dir / "per_round_log.csv"

    full_config = vars(args).copy()
    full_config.update(cfg)
    full_config["diagnostic_note"] = (
        "This is a stability diagnostic from a fixed checkpoint, not a Table IV formal result. "
        "Configs with algorithm_modification=true intentionally change the unlearning objective "
        "or optimizer strength to diagnose stability."
    )
    write_json(config_path, full_config)

    result = {
        "status": "started",
        "error_if_any": None,
        "name": cfg["name"],
        "algorithm_modification": cfg["algorithm_modification"],
        "objective": cfg["objective"],
        "seed": args.seed,
        "clean_acc_before": None,
        "ASR_before": None,
        "clean_acc_after": None,
        "ASR_after": None,
        "ASR_after_percent": None,
        "clean_accuracy_collapsed": None,
        "target_client_size": common["target_client_size"],
        "poison_num": common["poison_num"],
        "poison_num_expected": common["poison_num_expected"],
        "elapsed_seconds": None,
        "config_path": str(config_path),
        "result_path": str(result_path),
        "per_round_log": str(log_path),
    }
    try:
        set_seed(args.seed)
        device = common["device"]
        model = build_model(args.model_type, args.dataset, common["n_classes"]).to(device)
        model.load_state_dict(load_state_dict(Path(args.checkpoint), device))
        reference_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        optimizer = optim.SGD(model.parameters(), lr=cfg["unlearn_lr"], momentum=cfg["momentum"])
        criterion = nn.CrossEntropyLoss()

        clean_before, asr_before = evaluate_pair(
            model,
            common["clean_eval_loader"],
            common["trigger_eval_loader"],
            args.target_class,
            device,
        )
        result["clean_acc_before"] = clean_before
        result["ASR_before"] = asr_before

        retain_iter = iter(common["retain_loader"])
        for round_idx in range(0, cfg["rounds"] + 1):
            if round_idx == 0:
                clean_acc, asr = clean_before, asr_before
                row = {
                    "round": 0,
                    "clean_acc": clean_acc,
                    "ASR": asr,
                    "ASR_percent": asr * 100.0,
                    "unlearn_loss": "",
                    "forget_ce": "",
                    "retain_ce": "",
                    "proximal_loss": "",
                    "objective_loss": "",
                    "grad_l2_norm": "",
                    "update_l2_norm": "",
                    "grad_clip": cfg["grad_clip"] if cfg["grad_clip"] is not None else "",
                    "clipped_grad_norm_before": "",
                    "elapsed_seconds": time.time() - start,
                }
                append_csv(log_path, PER_ROUND_FIELDS, row)
                continue

            before_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.train()
            total_forget_ce = 0.0
            total_retain_ce = 0.0
            total_prox = 0.0
            total_objective = 0.0
            total_grad_norm_sq = 0.0
            total_clip_before_sq = 0.0
            num_steps = 0

            for forget_x, forget_y in common["forget_loader"]:
                forget_x = forget_x.to(device)
                forget_y = forget_y.to(device)
                optimizer.zero_grad(set_to_none=True)

                forget_ce = criterion(model(forget_x), forget_y)
                objective = -forget_ce
                retain_ce_value = torch.tensor(0.0, device=device)
                prox_value = torch.tensor(0.0, device=device)

                if cfg["retain_reg"]:
                    (retain_x, retain_y), retain_iter = next_retain_batch(retain_iter, common["retain_loader"])
                    retain_x = retain_x.to(device)
                    retain_y = retain_y.to(device)
                    retain_ce_value = criterion(model(retain_x), retain_y)
                    objective = objective + cfg["lambda_retain"] * retain_ce_value

                if cfg["proximal_mu"] > 0:
                    prox_value = 0.5 * cfg["proximal_mu"] * model_distance_loss(model, reference_state, device)
                    objective = objective + prox_value

                objective.backward()
                grad_norm_before = grad_l2_norm(model.parameters())
                if cfg["grad_clip"] is not None:
                    clip_before = float(clip_grad_norm_(model.parameters(), max_norm=float(cfg["grad_clip"])))
                else:
                    clip_before = grad_norm_before
                grad_norm_after = grad_l2_norm(model.parameters())
                optimizer.step()

                total_forget_ce += float(forget_ce.item())
                total_retain_ce += float(retain_ce_value.item())
                total_prox += float(prox_value.item())
                total_objective += float(objective.item())
                total_grad_norm_sq += grad_norm_after ** 2
                total_clip_before_sq += clip_before ** 2
                num_steps += 1

            after_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            clean_acc, asr = evaluate_pair(
                model,
                common["clean_eval_loader"],
                common["trigger_eval_loader"],
                args.target_class,
                device,
            )
            append_csv(
                log_path,
                PER_ROUND_FIELDS,
                {
                    "round": round_idx,
                    "clean_acc": clean_acc,
                    "ASR": asr,
                    "ASR_percent": asr * 100.0,
                    "unlearn_loss": total_objective / max(num_steps, 1),
                    "forget_ce": total_forget_ce / max(num_steps, 1),
                    "retain_ce": total_retain_ce / max(num_steps, 1) if cfg["retain_reg"] else "",
                    "proximal_loss": total_prox / max(num_steps, 1) if cfg["proximal_mu"] > 0 else "",
                    "objective_loss": total_objective / max(num_steps, 1),
                    "grad_l2_norm": math.sqrt(total_grad_norm_sq),
                    "update_l2_norm": state_l2_delta(before_state, after_state),
                    "grad_clip": cfg["grad_clip"] if cfg["grad_clip"] is not None else "",
                    "clipped_grad_norm_before": math.sqrt(total_clip_before_sq),
                    "elapsed_seconds": time.time() - start,
                },
            )

        final_clean, final_asr = evaluate_pair(
            model,
            common["clean_eval_loader"],
            common["trigger_eval_loader"],
            args.target_class,
            device,
        )
        result["clean_acc_after"] = final_clean
        result["ASR_after"] = final_asr
        result["ASR_after_percent"] = final_asr * 100.0
        result["clean_accuracy_collapsed"] = final_clean < args.collapse_clean_acc_threshold
        result["status"] = "ok"
        torch.save(model.state_dict(), config_dir / "unlearned_checkpoint.pth")
    except Exception as exc:
        result["status"] = "error"
        result["error_if_any"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    finally:
        result["elapsed_seconds"] = time.time() - start
        write_json(result_path, result)
    return result


def write_summary(output_dir: Path, rows: List[Dict]) -> None:
    write_json(output_dir / "grid_summary.json", {"results": rows})
    with (output_dir / "grid_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "name",
            "status",
            "algorithm_modification",
            "objective",
            "clean_acc_before",
            "ASR_before",
            "clean_acc_after",
            "ASR_after",
            "ASR_after_percent",
            "clean_accuracy_collapsed",
            "target_client_size",
            "poison_num",
            "elapsed_seconds",
            "result_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


PER_ROUND_FIELDS = [
    "round",
    "clean_acc",
    "ASR",
    "ASR_percent",
    "unlearn_loss",
    "forget_ce",
    "retain_ce",
    "proximal_loss",
    "objective_loss",
    "grad_l2_norm",
    "update_l2_norm",
    "grad_clip",
    "clipped_grad_norm_before",
    "elapsed_seconds",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Unlearning stability grid from a fixed Table IV checkpoint.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100", "mnist"])
    parser.add_argument("--n_client", type=int, default=50)
    parser.add_argument("--n_server1", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dirichlet_alpha", type=float, default=1.0)
    parser.add_argument("--model_type", default="cifar10cnn_v2", choices=["lenet5", "cifar10cnn_v2", "resnet18"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target_client", type=int, default=0)
    parser.add_argument("--target_class", type=int, default=9)
    parser.add_argument("--poison_ratio", type=float, default=0.8)
    parser.add_argument("--trigger_size", type=int, default=3)
    parser.add_argument("--trigger_location", default="bottom-right")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use_augmentation", action="store_true")
    parser.add_argument("--collapse_clean_acc_threshold", type=float, default=0.50)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise SystemExit(f"Refusing to write into non-empty output_dir: {output_dir}")

    if not Path(args.checkpoint).exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    set_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")

    fed_data = FedDataLoader(
        args.dataset,
        batch_size=args.batch_size,
        n_clients=args.n_client,
        alpha=args.dirichlet_alpha,
        seed=args.seed,
        use_augmentation=args.use_augmentation,
    )
    target_subset = Subset(fed_data.train_dataset, fed_data.client_train_indices[args.target_client])
    forget_dataset = ExactPoisonedDataset(
        target_subset,
        target_class=args.target_class,
        poison_ratio=args.poison_ratio,
        seed=args.seed,
        trigger_size=args.trigger_size,
        trigger_location=args.trigger_location,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed + 100000 + args.target_client)
    forget_loader = DataLoader(
        forget_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    retain_loader = build_retain_loader(args, fed_data, args.target_client)
    clean_eval_loader, trigger_eval_loader = make_eval_loaders(fed_data, args)
    target_client_size = len(forget_dataset)
    poison_num = forget_dataset.poison_num
    poison_num_expected = int(math.floor(args.poison_ratio * target_client_size))

    common = {
        "device": device,
        "n_classes": fed_data.n_classes,
        "forget_loader": forget_loader,
        "retain_loader": retain_loader,
        "clean_eval_loader": clean_eval_loader,
        "trigger_eval_loader": trigger_eval_loader,
        "target_client_size": target_client_size,
        "poison_num": poison_num,
        "poison_num_expected": poison_num_expected,
    }

    top_config = vars(args).copy()
    top_config.update(
        {
            "target_client_size": target_client_size,
            "poison_num": poison_num,
            "poison_num_expected": poison_num_expected,
            "target_class_source": "unconfirmed_assumption_from_reference_CWT_code",
            "grid_size": len(build_grid()),
        }
    )
    write_json(output_dir / "grid_config.json", top_config)

    rows = []
    for cfg in build_grid():
        result = run_config(args, cfg, common)
        rows.append(result)
        write_summary(output_dir, rows)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if result.get("status") != "ok":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
