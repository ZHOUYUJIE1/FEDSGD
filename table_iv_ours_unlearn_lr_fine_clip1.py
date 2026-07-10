import argparse
import csv
import json
import math
import random
import time
import traceback
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset

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
    "unlearn_loss",
    "grad_l2_norm",
    "update_l2_norm",
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
    trigger_loader = DataLoader(triggered, batch_size=args.eval_batch_size, shuffle=False)
    return clean_loader, trigger_loader


def evaluate_pair(model, clean_loader, trigger_loader, args, device):
    clean_acc = evaluate_accuracy(model, clean_loader, device)
    asr = evaluate_asr(model, trigger_loader, args.target_class, device)
    return clean_acc, asr


def run_lr(args, lr: float, common: Dict) -> Dict:
    start = time.time()
    output_dir = Path(args.output_dir).resolve()
    lr_id = f"lr_{lr:.5f}".replace(".", "p")
    lr_dir = output_dir / lr_id
    if lr_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing directory: {lr_dir}")
    lr_dir.mkdir(parents=True, exist_ok=False)

    config_path = lr_dir / "config.json"
    result_path = lr_dir / "result.json"
    log_path = lr_dir / "per_round_log.csv"

    config = vars(args).copy()
    config.update(
        {
            "unlearn_lr": lr,
            "momentum": 0.0,
            "grad_clip": 1.0,
            "proximal_mu": 0.0,
            "retain_reg": False,
            "lambda_retain": 0.0,
            "rounds": 10,
            "objective": "-CE(forget_poisoned) with grad_clip=1",
            "algorithm_modification": True,
            "diagnostic_note": "Fine-grained lr diagnostic from fixed checkpoint; not a Table IV formal result.",
        }
    )
    write_json(config_path, config)

    result = {
        "status": "started",
        "error_if_any": None,
        "config_id": lr_id,
        "unlearn_lr": lr,
        "momentum": 0.0,
        "grad_clip": 1.0,
        "seed": args.seed,
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
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.0)
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
                "unlearn_loss": "",
                "grad_l2_norm": "",
                "update_l2_norm": "",
                "elapsed_seconds": time.time() - start,
            },
        )

        for round_idx in range(1, args.rounds + 1):
            before_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.train()
            total_objective = 0.0
            total_grad_sq = 0.0
            steps = 0
            for data, target in common["forget_loader"]:
                data = data.to(device)
                target = target.to(device)
                optimizer.zero_grad(set_to_none=True)
                forget_ce = criterion(model(data), target)
                objective = -forget_ce
                objective.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                gn = grad_l2_norm(model.parameters())
                optimizer.step()
                total_objective += float(objective.item())
                total_grad_sq += gn ** 2
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
                    "unlearn_loss": total_objective / max(steps, 1),
                    "grad_l2_norm": math.sqrt(total_grad_sq),
                    "update_l2_norm": state_l2_delta(before_state, after_state),
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
        torch.save(model.state_dict(), lr_dir / "unlearned_checkpoint.pth")
    except Exception as exc:
        result["status"] = "error"
        result["error_if_any"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    finally:
        result["elapsed_seconds"] = time.time() - start
        write_json(result_path, result)
    return result


def write_summary(output_dir: Path, rows: List[Dict]) -> None:
    write_json(output_dir / "fine_summary.json", {"results": rows})
    fields = [
        "config_id",
        "status",
        "unlearn_lr",
        "momentum",
        "grad_clip",
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
    with (output_dir / "fine_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def parse_lrs(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-grained lr diagnostic for clip=1 unlearning.")
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
    parser.add_argument("--lr_list", default="0.00055,0.00060,0.00065,0.00070,0.00075,0.00080,0.00085,0.00090,0.00095")
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
        }
    )
    write_json(output_dir / "fine_config.json", top_config)

    common = {
        "device": device,
        "n_classes": fed_data.n_classes,
        "forget_loader": forget_loader,
        "clean_eval_loader": clean_eval_loader,
        "trigger_eval_loader": trigger_eval_loader,
        "target_client_size": target_client_size,
        "poison_num": poison_num,
        "poison_num_expected": poison_num_expected,
    }
    rows = []
    for lr in parse_lrs(args.lr_list):
        result = run_lr(args, lr, common)
        rows.append(result)
        write_summary(output_dir, rows)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if result.get("status") != "ok":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
