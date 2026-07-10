import argparse
import csv
import json
import math
import random
import time
import traceback
from pathlib import Path
from typing import Dict, Iterable

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
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def append_csv(path: Path, fieldnames, row: Dict) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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


def run(args):
    start = time.time()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    required_absent = [
        output_dir / "config.json",
        output_dir / "result.json",
        output_dir / "per_round_log.csv",
        output_dir / "summary.csv",
        output_dir / "unlearned_checkpoint.pth",
    ]
    existing = [str(p) for p in required_absent if p.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing output files: {existing}")

    config = vars(args).copy()
    config.update(
        {
            "retain_reg": True,
            "proximal_mu": 0.0,
            "objective": "-CE(forget_poisoned) + lambda_retain * CE(retain_clean)",
            "algorithm_modification": True,
            "protocol_note": "Revised/corrected retain-protection unlearning protocol; not original-code reproduction.",
            "target_class_source": "unconfirmed_assumption_from_reference_CWT_code",
        }
    )
    write_json(output_dir / "config.json", config)

    result = {
        "status": "started",
        "error_if_any": None,
        "clean_acc_before": None,
        "ASR_before": None,
        "clean_acc_after": None,
        "ASR_after": None,
        "ASR_percent_after": None,
        "clean_drop_pp": None,
        "ASR_drop_pp": None,
        "target_client_size": None,
        "poison_num": None,
        "target_class": args.target_class,
        "poison_ratio": args.poison_ratio,
        "trigger_size": args.trigger_size,
        "trigger_location": args.trigger_location,
        "unlearn_lr": args.unlearn_lr,
        "grad_clip": args.grad_clip,
        "lambda_retain": args.lambda_retain,
        "momentum": args.momentum,
        "unlearning_rounds": args.unlearning_rounds,
        "elapsed_seconds": None,
    }

    try:
        set_seed(args.seed)
        device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        if device.type != "cuda":
            raise RuntimeError("CUDA is required for confirmation; refusing to run on CPU")

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

        model = cifar10cnn_v2(n_class=fed_data.n_classes, in_dim=3).to(device)
        model.load_state_dict(load_state_dict(Path(args.checkpoint), device))
        optimizer = optim.SGD(model.parameters(), lr=args.unlearn_lr, momentum=args.momentum)
        criterion = nn.CrossEntropyLoss()

        clean_before, asr_before = evaluate_pair(model, clean_eval_loader, trigger_eval_loader, args, device)
        result["clean_acc_before"] = clean_before
        result["ASR_before"] = asr_before
        result["target_client_size"] = len(forget_dataset)
        result["poison_num"] = forget_dataset.poison_num

        append_csv(
            output_dir / "per_round_log.csv",
            PER_ROUND_FIELDS,
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

        retain_iter = iter(retain_loader)
        for round_idx in range(1, args.unlearning_rounds + 1):
            before_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            total_forget = 0.0
            total_retain = 0.0
            total_loss = 0.0
            total_grad_sq = 0.0
            steps = 0
            model.train()
            for forget_x, forget_y in forget_loader:
                forget_x = forget_x.to(device)
                forget_y = forget_y.to(device)
                (retain_x, retain_y), retain_iter = next_retain_batch(retain_iter, retain_loader)
                retain_x = retain_x.to(device)
                retain_y = retain_y.to(device)

                optimizer.zero_grad(set_to_none=True)
                forget_ce = criterion(model(forget_x), forget_y)
                retain_ce = criterion(model(retain_x), retain_y)
                loss = -forget_ce + args.lambda_retain * retain_ce
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                grad_after_clip = grad_l2_norm(model.parameters())
                optimizer.step()

                total_forget += float(forget_ce.item())
                total_retain += float(retain_ce.item())
                total_loss += float(loss.item())
                total_grad_sq += grad_after_clip ** 2
                steps += 1

            after_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            clean_acc, asr = evaluate_pair(model, clean_eval_loader, trigger_eval_loader, args, device)
            append_csv(
                output_dir / "per_round_log.csv",
                PER_ROUND_FIELDS,
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

        clean_after, asr_after = evaluate_pair(model, clean_eval_loader, trigger_eval_loader, args, device)
        torch.save(model.state_dict(), output_dir / "unlearned_checkpoint.pth")
        result["clean_acc_after"] = clean_after
        result["ASR_after"] = asr_after
        result["ASR_percent_after"] = asr_after * 100.0
        result["clean_drop_pp"] = (clean_before - clean_after) * 100.0
        result["ASR_drop_pp"] = (asr_before - asr_after) * 100.0
        result["status"] = "ok"
    except Exception as exc:
        result["status"] = "error"
        result["error_if_any"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    finally:
        result["elapsed_seconds"] = time.time() - start
        write_json(output_dir / "result.json", result)
        with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            fields = [
                "status",
                "clean_acc_before",
                "ASR_before",
                "clean_acc_after",
                "ASR_after",
                "ASR_percent_after",
                "clean_drop_pp",
                "ASR_drop_pp",
                "target_client_size",
                "poison_num",
                "unlearn_lr",
                "grad_clip",
                "lambda_retain",
                "momentum",
                "unlearning_rounds",
                "elapsed_seconds",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerow({k: result.get(k) for k in fields})
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Confirm revised Table IV Ours retain-protection protocol for seed 0.")
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
    parser.add_argument("--unlearn_lr", type=float, default=0.0009)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--lambda_retain", type=float, default=5.0)
    parser.add_argument("--unlearning_rounds", type=int, default=10)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use_augmentation", action="store_true")
    return parser.parse_args()


def main():
    result = run(parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result.get("status") == "ok" else 1)


if __name__ == "__main__":
    main()
