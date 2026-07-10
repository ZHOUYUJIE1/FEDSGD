import argparse
import csv
import json
import math
import random
import time
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import ConcatDataset, DataLoader, Subset

from model.data import loader as FedDataLoader
from model.lenet import cifar10cnn_v2
from table_iv_backdoor_utils import (
    ExactPoisonedDataset,
    evaluate_accuracy,
    evaluate_asr,
    make_triggered_tensor_dataset,
)


PER_ROUND_FIELDS = [
    "round",
    "clean_acc",
    "ASR",
    "ASR_percent",
    "forget_ce",
    "retain_ce",
    "total_loss",
    "grad_l2_norm",
    "update_l2_norm",
    "clean_drop_pp",
    "ASR_drop_pp",
    "elapsed_seconds",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def append_csv(path: Path, row: Dict) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PER_ROUND_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_state_dict(path: Path, device: torch.device):
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj


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


def make_eval_loaders(fed_data, args):
    clean_loader = DataLoader(fed_data.test_dataset, batch_size=args.eval_batch_size, shuffle=False)
    triggered = make_triggered_tensor_dataset(
        fed_data.test_dataset,
        target_class=args.target_class,
        trigger_size=args.trigger_size,
        trigger_location=args.trigger_location,
        batch_size=args.eval_batch_size,
    )
    triggered_loader = DataLoader(triggered, batch_size=args.eval_batch_size, shuffle=False)
    return clean_loader, triggered_loader


def evaluate_pair(model, clean_loader, triggered_loader, args, device) -> Tuple[float, float]:
    clean_acc = evaluate_accuracy(model, clean_loader, device)
    asr = evaluate_asr(model, triggered_loader, args.target_class, device)
    return clean_acc, asr


def build_retain_loader(args, fed_data):
    datasets = []
    for client_id, indices in enumerate(fed_data.client_train_indices):
        if client_id == args.target_client:
            continue
        datasets.append(Subset(fed_data.train_dataset, indices))
    retain_dataset = ConcatDataset(datasets)
    generator = torch.Generator()
    generator.manual_seed(args.seed + 424242)
    return DataLoader(retain_dataset, batch_size=args.batch_size, shuffle=True, generator=generator)


def next_retain_batch(retain_iter, retain_loader):
    try:
        return next(retain_iter), retain_iter
    except StopIteration:
        retain_iter = iter(retain_loader)
        return next(retain_iter), retain_iter


def build_grid(args) -> List[Dict]:
    configs = []
    for lr in args.lr_values:
        for clip in args.grad_clip_values:
            for lam in args.lambda_retain_values:
                configs.append(
                    {
                        "config_id": f"retain_lr{lr:.4g}_clip{clip:g}_lam{lam:g}".replace(".", "p"),
                        "unlearn_lr": float(lr),
                        "momentum": 0.0,
                        "grad_clip": float(clip),
                        "lambda_retain": float(lam),
                        "proximal_mu": 0.0,
                        "retain_reg": True,
                        "rounds": args.rounds,
                        "objective": "-CE(forget_poisoned) + lambda_retain * CE(retain_clean)",
                        "algorithm_modification": True,
                    }
                )
    return configs


def run_config(args, cfg: Dict, common: Dict) -> Dict:
    start = time.time()
    output_dir = Path(args.output_dir).resolve()
    config_dir = output_dir / cfg["config_id"]
    if config_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing directory: {config_dir}")
    config_dir.mkdir(parents=True, exist_ok=False)

    config_path = config_dir / "config.json"
    result_path = config_dir / "result.json"
    log_path = config_dir / "per_round_log.csv"

    full_config = vars(args).copy()
    full_config.update(cfg)
    full_config["target_client_size"] = common["target_client_size"]
    full_config["poison_num"] = common["poison_num"]
    full_config["poison_num_expected"] = common["poison_num_expected"]
    full_config["target_class_source"] = "unconfirmed_assumption_from_reference_CWT_code"
    full_config["diagnostic_note"] = (
        "Retain-protection unlearning diagnostic from a fixed checkpoint. "
        "This is a revised/corrected protocol exploration, not an original Table IV reproduction."
    )
    write_json(config_path, full_config)

    result = {
        "status": "started",
        "error_if_any": None,
        "config_id": cfg["config_id"],
        "unlearn_lr": cfg["unlearn_lr"],
        "momentum": cfg["momentum"],
        "grad_clip": cfg["grad_clip"],
        "lambda_retain": cfg["lambda_retain"],
        "proximal_mu": cfg["proximal_mu"],
        "clean_acc_before": None,
        "ASR_before": None,
        "clean_acc_after": None,
        "ASR_after": None,
        "ASR_after_percent": None,
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
        model = cifar10cnn_v2(n_class=common["n_classes"], in_dim=3).to(device)
        model.load_state_dict(load_state_dict(Path(args.checkpoint), device))
        optimizer = optim.SGD(model.parameters(), lr=cfg["unlearn_lr"], momentum=0.0)
        criterion = nn.CrossEntropyLoss()

        clean_before, asr_before = evaluate_pair(
            model, common["clean_eval_loader"], common["trigger_eval_loader"], args, device
        )
        result["clean_acc_before"] = clean_before
        result["ASR_before"] = asr_before

        append_csv(
            log_path,
            {
                "round": 0,
                "clean_acc": clean_before,
                "ASR": asr_before,
                "ASR_percent": asr_before * 100.0,
                "forget_ce": "",
                "retain_ce": "",
                "total_loss": "",
                "grad_l2_norm": "",
                "update_l2_norm": "",
                "clean_drop_pp": 0.0,
                "ASR_drop_pp": 0.0,
                "elapsed_seconds": time.time() - start,
            },
        )

        retain_iter = iter(common["retain_loader"])
        for round_idx in range(1, cfg["rounds"] + 1):
            before_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.train()
            total_forget = 0.0
            total_retain = 0.0
            total_loss = 0.0
            total_grad_sq = 0.0
            steps = 0

            for forget_x, forget_y in common["forget_loader"]:
                forget_x = forget_x.to(device)
                forget_y = forget_y.to(device)
                (retain_x, retain_y), retain_iter = next_retain_batch(retain_iter, common["retain_loader"])
                retain_x = retain_x.to(device)
                retain_y = retain_y.to(device)

                optimizer.zero_grad(set_to_none=True)
                forget_ce = criterion(model(forget_x), forget_y)
                retain_ce = criterion(model(retain_x), retain_y)
                loss = -forget_ce + cfg["lambda_retain"] * retain_ce
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=cfg["grad_clip"])
                grad_after_clip = grad_l2_norm(model.parameters())
                optimizer.step()

                total_forget += float(forget_ce.item())
                total_retain += float(retain_ce.item())
                total_loss += float(loss.item())
                total_grad_sq += grad_after_clip ** 2
                steps += 1

            after_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            clean_acc, asr = evaluate_pair(
                model, common["clean_eval_loader"], common["trigger_eval_loader"], args, device
            )
            append_csv(
                log_path,
                {
                    "round": round_idx,
                    "clean_acc": clean_acc,
                    "ASR": asr,
                    "ASR_percent": asr * 100.0,
                    "forget_ce": total_forget / max(steps, 1),
                    "retain_ce": total_retain / max(steps, 1),
                    "total_loss": total_loss / max(steps, 1),
                    "grad_l2_norm": math.sqrt(total_grad_sq),
                    "update_l2_norm": state_l2_delta(before_state, after_state),
                    "clean_drop_pp": (clean_before - clean_acc) * 100.0,
                    "ASR_drop_pp": (asr_before - asr) * 100.0,
                    "elapsed_seconds": time.time() - start,
                },
            )

        clean_after, asr_after = evaluate_pair(
            model, common["clean_eval_loader"], common["trigger_eval_loader"], args, device
        )
        result["clean_acc_after"] = clean_after
        result["ASR_after"] = asr_after
        result["ASR_after_percent"] = asr_after * 100.0
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
    write_json(output_dir / "retain_grid_summary.json", {"results": rows})
    fields = [
        "config_id",
        "status",
        "unlearn_lr",
        "momentum",
        "grad_clip",
        "lambda_retain",
        "proximal_mu",
        "clean_acc_before",
        "ASR_before",
        "clean_acc_after",
        "ASR_after",
        "ASR_after_percent",
        "target_client_size",
        "poison_num",
        "elapsed_seconds",
        "result_path",
    ]
    with (output_dir / "retain_grid_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def parse_float_list(text: str) -> List[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Retain-protection unlearning diagnostic grid.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10"])
    parser.add_argument("--n_client", type=int, default=50)
    parser.add_argument("--n_server1", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dirichlet_alpha", type=float, default=1.0)
    parser.add_argument("--model_type", default="cifar10cnn_v2", choices=["cifar10cnn_v2"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target_client", type=int, default=0)
    parser.add_argument("--target_class", type=int, default=9)
    parser.add_argument("--poison_ratio", type=float, default=0.8)
    parser.add_argument("--trigger_size", type=int, default=3)
    parser.add_argument("--trigger_location", default="bottom-right")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--lr_values", type=parse_float_list, default=parse_float_list("0.0005,0.0007,0.0009"))
    parser.add_argument("--grad_clip_values", type=parse_float_list, default=parse_float_list("1,2,5"))
    parser.add_argument("--lambda_retain_values", type=parse_float_list, default=parse_float_list("0.5,1,2,5,10"))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use_augmentation", action="store_true")
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
    if device.type != "cuda":
        raise SystemExit("CUDA is required for this diagnostic; refusing to run on CPU")

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
    forget_loader = DataLoader(forget_dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    retain_loader = build_retain_loader(args, fed_data)
    clean_eval_loader, trigger_eval_loader = make_eval_loaders(fed_data, args)
    target_client_size = len(forget_dataset)
    poison_num = forget_dataset.poison_num
    poison_num_expected = int(math.floor(args.poison_ratio * target_client_size))

    top_config = vars(args).copy()
    top_config.update(
        {
            "target_client_size": target_client_size,
            "poison_num": poison_num,
            "poison_num_expected": poison_num_expected,
            "target_class_source": "unconfirmed_assumption_from_reference_CWT_code",
            "grid_size": len(build_grid(args)),
            "retain_data_source": "ConcatDataset of all non-target clients' clean train subsets",
            "retain_batches_per_forget_epoch": "one retain batch for each forget batch step",
            "clipping_scope": "combined gradient after backward on -forget_ce + lambda_retain * retain_ce",
        }
    )
    write_json(output_dir / "retain_grid_config.json", top_config)

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

    rows = []
    for cfg in build_grid(args):
        result = run_config(args, cfg, common)
        rows.append(result)
        write_summary(output_dir, rows)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if result.get("status") != "ok":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
