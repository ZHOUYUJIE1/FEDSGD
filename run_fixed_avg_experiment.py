import argparse
import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from model.data import loader
from model.lenet import cifar10cnn_v2, lenet5, resnet18
from topk import TopkCompressor


RESULT_DIR = Path("/home/kemove/zlf/FedSGD2/results/fixed_avg")
ADAPTIVE_LOG = RESULT_DIR / "adaptive_retained_ratio_log.csv"
RHO_JSON = RESULT_DIR / "rho_avg.json"
SUMMARY_CSV = RESULT_DIR / "fixed_avg_summary.csv"
TABLE_MD = RESULT_DIR / "table_vi_for_paper.md"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_model(model_type: str, n_class: int, in_dim: int, device: torch.device) -> nn.Module:
    if model_type == "resnet18":
        return resnet18(n_class=n_class, in_dim=in_dim).to(device)
    if model_type == "cifar10cnn_v2":
        return cifar10cnn_v2(n_class=n_class, in_dim=in_dim).to(device)
    return lenet5(n_class=n_class, in_dim=in_dim).to(device)


def infer_model_type(dataset_name: str) -> Tuple[str, int]:
    if dataset_name == "cifar100":
        return "resnet18", 3
    if dataset_name == "cifar10":
        return "cifar10cnn_v2", 3
    return "lenet5", 1


def make_optimizer(model: nn.Module, model_type: str) -> optim.Optimizer:
    if model_type == "resnet18":
        return optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    return optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


def compute_normalized_direction(gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    normalized = {}
    for name, grad in gradients.items():
        norm = torch.norm(grad.flatten(), p=2)
        normalized[name] = grad / norm if norm > 1e-10 else torch.zeros_like(grad)
    return normalized


def estimate_topk_bits(k: int, d: int, value_bits: int = 32) -> int:
    index_bits = math.ceil(math.log2(max(d, 2)))
    return int(k * (value_bits + index_bits))


def compute_raw_grads(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int],
) -> Tuple[Dict[str, torch.Tensor], int, float, int]:
    criterion = nn.CrossEntropyLoss()
    model.train()
    grad_sums = {name: torch.zeros_like(param, device=device) for name, param in model.named_parameters()}
    n_samples = 0
    loss_sum = 0.0
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        data = data.to(device)
        target = target.to(device)
        batch_size = int(data.size(0))
        n_samples += batch_size

        output = model(data)
        loss = criterion(output, target)
        loss_sum += float(loss.item()) * batch_size
        correct += int(output.argmax(dim=1).eq(target).sum().item())

        model.zero_grad(set_to_none=True)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            if torch.isnan(grad).any() or torch.isinf(grad).any():
                grad = torch.where(torch.isnan(grad) | torch.isinf(grad), torch.zeros_like(grad), grad)
            grad_sums[name] += grad * batch_size

    denom = max(n_samples, 1)
    return {name: grad / denom for name, grad in grad_sums.items()}, n_samples, loss_sum, correct


def sparsify_grads(
    raw_grads: Dict[str, torch.Tensor],
    compressor: TopkCompressor,
    normalized_direction: Optional[Dict[str, torch.Tensor]],
    method: str,
    epoch: int,
    client_id: int,
    log_writer: Optional[csv.DictWriter],
    counters: Dict[str, float],
) -> Dict[str, torch.Tensor]:
    sparse_grads = {}
    for layer_name, local_grad in raw_grads.items():
        global_grad = None
        if normalized_direction is not None and layer_name in normalized_direction:
            global_grad = normalized_direction[layer_name]
        values, indices = compressor.compress_tensor(local_grad, global_gradient=global_grad)
        d = int(local_grad.numel())
        k = int(values.numel())
        counters["sum_k"] += k
        counters["sum_d"] += d
        counters["n_records"] += 1
        counters["comm_bits"] += estimate_topk_bits(k, d)
        if log_writer is not None:
            log_writer.writerow(
                {
                    "round": epoch,
                    "client_id": client_id,
                    "layer_name": layer_name,
                    "d": d,
                    "k": k,
                    "retained_ratio": k / d if d else 0.0,
                    "compression_mode": method,
                }
            )
        sparse_grads[layer_name] = compressor.decompress_tensor(values, indices, local_grad.shape)
    return sparse_grads


def aggregate_client_grads(
    client_grads: Iterable[Tuple[Dict[str, torch.Tensor], int]]
) -> Dict[str, torch.Tensor]:
    total_samples = 0
    total_grads: Dict[str, torch.Tensor] = {}
    for grads, n_samples in client_grads:
        total_samples += n_samples
        for name, grad in grads.items():
            if name not in total_grads:
                total_grads[name] = grad.detach().clone() * n_samples
            else:
                total_grads[name] += grad.detach() * n_samples
    denom = max(total_samples, 1)
    return {name: grad / denom for name, grad in total_grads.items()}


def apply_gradients(model: nn.Module, optimizer: optim.Optimizer, gradients: Dict[str, torch.Tensor]) -> None:
    optimizer.zero_grad(set_to_none=True)
    for name, param in model.named_parameters():
        param.grad = gradients.get(name, torch.zeros_like(param.data)).to(param.data.device)
    optimizer.step()


@torch.no_grad()
def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        correct += int(output.argmax(dim=1).eq(target).sum().item())
        total += int(target.numel())
    return correct / max(total, 1)


def run_method(args, method: str, fixed_retained_ratio: Optional[float] = None):
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    data = loader(
        args.dataset,
        batch_size=args.batch_size,
        n_clients=args.n_client,
        alpha=args.dirichlet_alpha,
        seed=args.seed,
        use_augmentation=False,
    )
    model_type, in_dim = infer_model_type(args.dataset)
    model = build_model(model_type, data.n_classes, in_dim, device)
    optimizer = make_optimizer(model, model_type)
    full_test_loader = DataLoader(data.test_dataset, batch_size=args.batch_size, shuffle=False)

    if method == "fixed":
        if fixed_retained_ratio is None:
            raise ValueError("fixed_retained_ratio is required for Fixed-Avg")
        compress_ratio = 1.0 - float(fixed_retained_ratio)
    else:
        compress_ratio = args.fallback_retained_ratio

    compressors = [
        TopkCompressor(
            compress_ratio=compress_ratio,
            min_sparsity_ratio=args.min_retained_ratio,
            max_sparsity_ratio=args.max_retained_ratio,
            compression_mode=method,
        )
        for _ in range(args.n_client)
    ]

    counters = {"sum_k": 0.0, "sum_d": 0.0, "n_records": 0.0, "comm_bits": 0.0}
    normalized_direction = None
    final_accuracy = 0.0
    log_file = None
    log_writer = None

    if method == "adaptive":
        log_file = ADAPTIVE_LOG.open("w", newline="", encoding="utf-8")
        log_writer = csv.DictWriter(
            log_file,
            fieldnames=["round", "client_id", "layer_name", "d", "k", "retained_ratio", "compression_mode"],
        )
        log_writer.writeheader()

    try:
        for epoch in range(1, args.n_epoch + 1):
            compressed_this_round = args.compress_first_round or epoch > 1
            client_payloads = []
            for client_id in range(args.n_client):
                train_loader, _ = data.get_loader(client_id, use_global_test=True)
                raw_grads, n_samples, _, _ = compute_raw_grads(
                    model,
                    train_loader,
                    device,
                    args.max_batches_per_client,
                )
                if compressed_this_round and method != "none":
                    sparse_grads = sparsify_grads(
                        raw_grads=raw_grads,
                        compressor=compressors[client_id],
                        normalized_direction=normalized_direction,
                        method=method,
                        epoch=epoch,
                        client_id=client_id,
                        log_writer=log_writer,
                        counters=counters,
                    )
                    client_payloads.append((sparse_grads, n_samples))
                else:
                    client_payloads.append((raw_grads, n_samples))
            gradients = aggregate_client_grads(client_payloads)
            normalized_direction = compute_normalized_direction(gradients)
            apply_gradients(model, optimizer, gradients)
            final_accuracy = evaluate(model, full_test_loader, device)
            print(f"[{method}] epoch={epoch}/{args.n_epoch} accuracy={final_accuracy:.6f}")
    finally:
        if log_file is not None:
            log_file.close()

    retained = counters["sum_k"] / counters["sum_d"] if counters["sum_d"] else 1.0
    return {
        "method": method,
        "avg_retained_ratio": retained,
        "internal_compress_ratio": compress_ratio if method == "fixed" else "adaptive",
        "final_accuracy": final_accuracy,
        "communication_bits": int(counters["comm_bits"]),
        "communication_mb": counters["comm_bits"] / 8.0 / 1024.0 / 1024.0,
        "sum_k": int(counters["sum_k"]),
        "sum_d": int(counters["sum_d"]),
        "n_records": int(counters["n_records"]),
        "model_type": model_type,
        "device": str(device),
    }


def write_outputs(args, adaptive_result, fixed_result) -> None:
    rho = adaptive_result["avg_retained_ratio"]
    rho_payload = {
        "rho_avg": rho,
        "sum_k": adaptive_result["sum_k"],
        "sum_d": adaptive_result["sum_d"],
        "n_records": adaptive_result["n_records"],
        "dataset": args.dataset,
        "dirichlet_alpha": args.dirichlet_alpha,
        "seed": args.seed,
        "n_client": args.n_client,
        "n_server1": args.n_server1,
        "n_epoch": args.n_epoch,
        "weighted_average": "rho_avg = sum(k) / sum(d)",
        "notes": "Only compressed adaptive rounds contribute to rho_avg; with compress_first_round=False, round 1 is warm-up and not logged.",
    }
    RHO_JSON.write_text(json.dumps(rho_payload, indent=2), encoding="utf-8")

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Method",
                "Avg retained ratio",
                "Internal compress_ratio",
                "Final accuracy",
                "Communication volume",
                "Notes",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "Method": "Ours adaptive",
                "Avg retained ratio": f"{adaptive_result['avg_retained_ratio']:.8f}",
                "Internal compress_ratio": "adaptive",
                "Final accuracy": f"{adaptive_result['final_accuracy']:.8f}",
                "Communication volume": f"{adaptive_result['communication_mb']:.6f} MB estimated top-k",
                "Notes": "weighted rho_avg=sum(k)/sum(d)",
            }
        )
        writer.writerow(
            {
                "Method": "Fixed-Avg",
                "Avg retained ratio": f"{fixed_result['avg_retained_ratio']:.8f}",
                "Internal compress_ratio": f"{fixed_result['internal_compress_ratio']:.8f}",
                "Final accuracy": f"{fixed_result['final_accuracy']:.8f}",
                "Communication volume": f"{fixed_result['communication_mb']:.6f} MB estimated top-k",
                "Notes": "fixed compress_ratio = 1 - adaptive rho_avg",
            }
        )

    table = f"""TABLE VI: Fair comparison with Fixed-Avg under the same average retained ratio on CIFAR-10.

| Method | Avg. retained ratio | Internal compress ratio | Final accuracy | Communication volume |
|---|---:|---:|---:|---:|
| Ours adaptive | {adaptive_result['avg_retained_ratio']:.6f} | adaptive | {adaptive_result['final_accuracy']:.6f} | {adaptive_result['communication_mb']:.6f} MB estimated top-k |
| Fixed-Avg | {fixed_result['avg_retained_ratio']:.6f} | {fixed_result['internal_compress_ratio']:.6f} | {fixed_result['final_accuracy']:.6f} | {fixed_result['communication_mb']:.6f} MB estimated top-k |
"""
    TABLE_MD.write_text(table, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Run adaptive rho logging and Fixed-Avg baseline for CIFAR-10.")
    parser.add_argument("--dataset", default="cifar10", choices=["mnist", "cifar10", "cifar100"])
    parser.add_argument("--n_client", type=int, default=50)
    parser.add_argument("--n_server1", type=int, default=5)
    parser.add_argument("--n_epoch", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dirichlet_alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_retained_ratio", type=float, default=0.3)
    parser.add_argument("--max_retained_ratio", type=float, default=0.8)
    parser.add_argument("--fallback_retained_ratio", type=float, default=0.1)
    parser.add_argument("--compress_first_round", action="store_true")
    parser.add_argument("--max_batches_per_client", type=int, default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir("/home/kemove/zlf/FedSGD2")

    adaptive_result = run_method(args, "adaptive")
    rho_avg = adaptive_result["avg_retained_ratio"]
    fixed_result = run_method(args, "fixed", fixed_retained_ratio=rho_avg)
    write_outputs(args, adaptive_result, fixed_result)

    print("\n[done] Fixed-Avg dry/full run completed")
    print(f"rho_avg={rho_avg:.8f}")
    print(f"fixed internal compress_ratio={fixed_result['internal_compress_ratio']:.8f}")
    print(f"results_dir={RESULT_DIR}")


if __name__ == "__main__":
    main()
