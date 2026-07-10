import argparse
import csv
import json
import math
import os
import random
import time
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

from model.data import loader as FedDataLoader
from model.lenet import cifar10cnn_v2, lenet5, resnet18
from table_iv_backdoor_utils import (
    ExactPoisonedDataset,
    evaluate_accuracy,
    evaluate_asr,
    make_triggered_tensor_dataset,
    mean_std,
)
from topk import TopkCompressor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def infer_model(dataset: str, model: str) -> Tuple[str, int]:
    if model != "auto":
        return model, 1 if dataset == "mnist" else 3
    if dataset == "cifar100":
        return "resnet18", 3
    if dataset == "cifar10":
        return "cifar10cnn_v2", 3
    return "lenet5", 1


def build_model(model_type: str, n_class: int, in_dim: int, device: torch.device) -> nn.Module:
    if model_type == "resnet18":
        return resnet18(n_class=n_class, in_dim=in_dim).to(device)
    if model_type == "cifar10cnn_v2":
        return cifar10cnn_v2(n_class=n_class, in_dim=in_dim).to(device)
    return lenet5(n_class=n_class, in_dim=in_dim).to(device)


def make_optimizer(model: nn.Module, model_type: str, lr: Optional[float]) -> optim.Optimizer:
    if lr is None:
        lr = 0.1 if model_type == "resnet18" else 0.01
    if model_type == "resnet18":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    return optim.SGD(model.parameters(), lr=lr, momentum=0.9)


def compute_normalized_direction(grads: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for name, grad in grads.items():
        norm = torch.linalg.vector_norm(grad.reshape(-1), ord=2)
        out[name] = grad / norm if float(norm.item()) > 1e-10 else torch.zeros_like(grad)
    return out


def compute_raw_grads(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    ascent: bool = False,
    max_batches: Optional[int] = None,
) -> Tuple[Dict[str, torch.Tensor], int, float, int]:
    criterion = nn.CrossEntropyLoss()
    model.train()
    grad_sums = {name: torch.zeros_like(param, device=device) for name, param in model.named_parameters()}
    sample_count = 0
    loss_sum = 0.0
    correct = 0

    for batch_idx, (images, labels) in enumerate(data_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)
        batch_size = int(labels.numel())
        sample_count += batch_size

        logits = model(images)
        loss = criterion(logits, labels)
        loss_sum += float(loss.item()) * batch_size
        correct += int(logits.argmax(dim=1).eq(labels).sum().item())

        model.zero_grad(set_to_none=True)
        loss.backward()
        sign = -1.0 if ascent else 1.0
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            if torch.isnan(grad).any() or torch.isinf(grad).any():
                grad = torch.where(torch.isnan(grad) | torch.isinf(grad), torch.zeros_like(grad), grad)
            grad_sums[name] += sign * grad * batch_size

    denom = max(sample_count, 1)
    return {name: grad / denom for name, grad in grad_sums.items()}, sample_count, loss_sum, correct


def sparsify_grads(
    raw_grads: Dict[str, torch.Tensor],
    compressor: TopkCompressor,
    normalized_direction: Optional[Dict[str, torch.Tensor]],
    unlearning_flag: bool,
) -> Dict[str, torch.Tensor]:
    sparse = {}
    for name, grad in raw_grads.items():
        global_grad = normalized_direction.get(name) if normalized_direction is not None else None
        values, indices = compressor.compress_tensor(
            grad,
            global_gradient=global_grad,
            unlearning_flag=unlearning_flag,
        )
        sparse[name] = compressor.decompress_tensor(values, indices, grad.shape)
    return sparse


def aggregate_weighted(payloads: Iterable[Tuple[Dict[str, torch.Tensor], int]]) -> Dict[str, torch.Tensor]:
    total = 0
    acc: Dict[str, torch.Tensor] = {}
    for grads, n_samples in payloads:
        total += int(n_samples)
        for name, grad in grads.items():
            if name not in acc:
                acc[name] = grad.detach().clone() * n_samples
            else:
                acc[name] += grad.detach() * n_samples
    denom = max(total, 1)
    return {name: grad / denom for name, grad in acc.items()}


def apply_gradients(model: nn.Module, optimizer: optim.Optimizer, grads: Dict[str, torch.Tensor]) -> None:
    optimizer.zero_grad(set_to_none=True)
    for name, param in model.named_parameters():
        param.grad = grads.get(name, torch.zeros_like(param.data)).to(param.data.device)
    optimizer.step()


def make_client_loaders(args, fed_data, target_client: int, poison_seed: int):
    train_loaders = []
    poisoned_dataset = None
    target_client_size = None
    poison_num = None
    for client_id in range(args.n_client):
        train_indices = fed_data.client_train_indices[client_id]
        train_dataset = Subset(fed_data.train_dataset, train_indices)
        if client_id == target_client:
            poisoned_dataset = ExactPoisonedDataset(
                train_dataset,
                target_class=args.target_class,
                poison_ratio=args.poison_ratio,
                seed=poison_seed,
                trigger_size=args.trigger_size,
                trigger_location=args.trigger_location,
            )
            target_client_size = len(poisoned_dataset)
            poison_num = poisoned_dataset.poison_num
            train_dataset = poisoned_dataset

        generator = torch.Generator()
        generator.manual_seed(args.seed * 100000 + client_id)
        train_loaders.append(
            DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                generator=generator,
            )
        )
    return train_loaders, poisoned_dataset, target_client_size, poison_num


def update_full_csvs(output_base: Path) -> None:
    result_paths = sorted(output_base.glob("seed_*/dry_run_result.json"))
    rows = []
    for result_path in result_paths:
        with result_path.open("r", encoding="utf-8") as f:
            result = json.load(f)
        if result.get("status") != "ok":
            continue
        rows.append(
            {
                "method": result.get("method"),
                "seed": result.get("seed"),
                "asr": result.get("ASR"),
                "clean_acc": result.get("clean_accuracy_if_available"),
                "target_class": result.get("target_class"),
                "target_client": result.get("target_client"),
                "target_client_size": result.get("target_client_size"),
                "poison_num": result.get("poison_num"),
                "config_path": str(result_path.parent / "config.json"),
                "result_path": str(result_path),
            }
        )

    if not rows:
        return

    raw_path = output_base / "ours_asr_raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "seed",
                "asr",
                "clean_acc",
                "target_class",
                "target_client",
                "target_client_size",
                "poison_num",
                "config_path",
                "result_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    asrs = [float(row["asr"]) for row in rows]
    mean_asr, std_asr = mean_std(asrs)
    summary_path = output_base / "ours_asr_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "mean_asr", "std_asr", "num_seeds"])
        writer.writeheader()
        writer.writerow(
            {
                "method": "Ours/adaptive",
                "mean_asr": mean_asr,
                "std_asr": std_asr,
                "num_seeds": len(asrs),
            }
        )


def run(args) -> Dict:
    start = time.time()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False if args.fail_if_output_exists else True)
    config_path = output_dir / "config.json"
    result_path = output_dir / "dry_run_result.json"

    target_client = args.target_client
    result = {
        "method": "Ours/adaptive",
        "seed": args.seed,
        "dataset": args.dataset,
        "n_client": args.n_client,
        "n_server1": args.n_server1,
        "batch_size": args.batch_size,
        "dirichlet_alpha": args.dirichlet_alpha,
        "model": None,
        "target_client": target_client,
        "target_client_size": None,
        "target_class": args.target_class,
        "target_class_source": "unconfirmed_assumption_from_reference_CWT_code",
        "poison_ratio": args.poison_ratio,
        "poison_num": None,
        "trigger_type": "white_pixel_patch",
        "trigger_size": args.trigger_size,
        "trigger_location": args.trigger_location,
        "unlearning_rounds": args.unlearning_rounds,
        "max_train_rounds": args.max_train_rounds,
        "clean_accuracy_if_available": None,
        "ASR": None,
        "status": "started",
        "error_if_any": None,
    }

    config = vars(args).copy()
    config["target_class_source"] = result["target_class_source"]
    config["notes"] = (
        "Independent Table IV Ours/adaptive backdoor ASR runner. "
        "Does not use or overwrite FedSGD2 cache. target_class=9 is an unconfirmed assumption."
    )
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        fed_data = FedDataLoader(
            args.dataset,
            batch_size=args.batch_size,
            n_clients=args.n_client,
            alpha=args.dirichlet_alpha,
            seed=args.seed,
            use_augmentation=False,
        )
        model_type, in_dim = infer_model(args.dataset, args.model)
        result["model"] = model_type
        model = build_model(model_type, fed_data.n_classes, in_dim, device)
        optimizer = make_optimizer(model, model_type, args.lr)
        compressor = TopkCompressor(
            compress_ratio=args.fallback_retained_ratio,
            min_sparsity_ratio=args.min_retained_ratio,
            max_sparsity_ratio=args.max_retained_ratio,
            bate=args.unlearn_bate,
            compression_mode="adaptive",
            unlearn_mode="unlearn",
        )

        train_loaders, _poisoned_dataset, target_size, poison_num = make_client_loaders(
            args,
            fed_data,
            target_client=target_client,
            poison_seed=args.seed,
        )
        result["target_client_size"] = target_size
        result["poison_num"] = poison_num

        normalized_direction = None
        for round_idx in range(1, args.max_train_rounds + 1):
            payloads = []
            use_compression = args.compress_first_round or round_idx > 1
            for client_id, train_loader in enumerate(train_loaders):
                raw_grads, n_samples, _loss, _correct = compute_raw_grads(
                    model,
                    train_loader,
                    device,
                    ascent=False,
                    max_batches=args.max_batches_per_client,
                )
                if use_compression:
                    raw_grads = sparsify_grads(
                        raw_grads,
                        compressor,
                        normalized_direction,
                        unlearning_flag=False,
                    )
                payloads.append((raw_grads, n_samples))
            global_grads = aggregate_weighted(payloads)
            normalized_direction = compute_normalized_direction(global_grads)
            apply_gradients(model, optimizer, global_grads)

        target_loader = train_loaders[target_client]
        for _round_idx in range(1, args.unlearning_rounds + 1):
            unlearn_grads, n_samples, _loss, _correct = compute_raw_grads(
                model,
                target_loader,
                device,
                ascent=True,
                max_batches=args.max_batches_per_client,
            )
            unlearn_grads = sparsify_grads(
                unlearn_grads,
                compressor,
                normalized_direction,
                unlearning_flag=True,
            )
            normalized_direction = compute_normalized_direction(unlearn_grads)
            apply_gradients(model, optimizer, unlearn_grads)

        clean_test_loader = DataLoader(
            fed_data.test_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        triggered_test = make_triggered_tensor_dataset(
            fed_data.test_dataset,
            target_class=args.target_class,
            trigger_size=args.trigger_size,
            trigger_location=args.trigger_location,
            batch_size=args.eval_batch_size,
        )
        triggered_loader = DataLoader(
            triggered_test,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        result["clean_accuracy_if_available"] = evaluate_accuracy(model, clean_test_loader, device)
        result["ASR"] = evaluate_asr(model, triggered_loader, args.target_class, device)
        result["status"] = "ok"
    except Exception as exc:
        result["status"] = "error"
        result["error_if_any"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    finally:
        result["elapsed_seconds"] = time.time() - start
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        if args.full_output_base:
            update_full_csvs(Path(args.full_output_base))

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Table IV Ours/adaptive CIFAR-10 backdoor ASR runner.")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100", "mnist"])
    parser.add_argument("--method", default="adaptive", choices=["adaptive"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n_client", type=int, default=8)
    parser.add_argument("--n_server1", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dirichlet_alpha", type=float, default=0.3)
    parser.add_argument("--model", default="auto", choices=["auto", "lenet5", "cifar10cnn_v2", "resnet18"])
    parser.add_argument("--target_client", type=int, default=0)
    parser.add_argument("--target_class", type=int, default=9)
    parser.add_argument("--poison_ratio", type=float, default=0.8)
    parser.add_argument("--trigger_size", type=int, default=3)
    parser.add_argument("--trigger_location", default="bottom-right")
    parser.add_argument("--unlearning_rounds", type=int, default=10)
    parser.add_argument("--max_train_rounds", type=int, default=1)
    parser.add_argument("--compress_first_round", action="store_true")
    parser.add_argument("--fallback_retained_ratio", type=float, default=0.1)
    parser.add_argument("--min_retained_ratio", type=float, default=0.1)
    parser.add_argument("--max_retained_ratio", type=float, default=0.9)
    parser.add_argument("--unlearn_bate", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--full_output_base", default=None)
    parser.add_argument("--fail_if_output_exists", action="store_true")
    parser.add_argument(
        "--max_batches_per_client",
        type=int,
        default=None,
        help="Optional debugging speed cap. Leave unset for declared experiments.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result.get("status") == "ok" else 1)


if __name__ == "__main__":
    main()
