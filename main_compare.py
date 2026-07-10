import logging
import os
import random

import torch
from torch.utils.data import DataLoader

from model.data import loader
from compare.server import server
from compare.server1 import server1
from compare.client import client
from model.logger import get_logger


def federated_learning():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger(log_dir="./log", log_name="fedsgd", level=logging.INFO)

    logger.info(f"使用设备: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU设备名称: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU数量: {torch.cuda.device_count()}")
    logger.info("-" * 60)

    # Shared encryption key (client + center). Used only by TopkCompressor encryption stage.
    ENCRYPTION_KEY = "fed_sgd_secret_key_2024"

    # ===== Experiment knobs =====
    n_client = 50
    n_server1 = 5
    n_epoch = 2000
    batch_size = 64

    # Data distribution (Dirichlet)
    dirichlet_alpha = 0.3

    # Dataset
    dataset_name = "cifar10"  # 'mnist' | 'cifar10' | 'cifar100'
    use_augmentation = False  # set True for normal training on CIFAR, set False for attack comparisons
    data_loader = loader(
        dataset_name,
        batch_size=batch_size,
        n_clients=n_client,
        alpha=dirichlet_alpha,
        seed=42,
        use_augmentation=use_augmentation,
    )

    # # Model type
    # if dataset_name in ("cifar10", "cifar100"):
    #     model_type = "resnet18"
    #     in_dim = 3
    # else:
    #     model_type = "lenet5"
    #     in_dim = 1
    if dataset_name == 'cifar100':
        model_type = 'resnet18'
        in_dim = 3
    elif dataset_name == 'cifar10':
        model_type = 'cifar10cnn_v2'
        in_dim = 3
    else:
        model_type = 'lenet5'
        in_dim = 1

    n_class = data_loader.n_classes

    # Compression settings
    compress_first_round = False
    # 'none' | 'fixed' | 'adaptive' | 'cate'
    compression_mode = "cate"

    # Fixed-batch mode for DLG-style attack comparisons (single-batch FedSGD)
    use_fixed_batch = False

    # Configure server-side attackers / unlearning if you want (kept minimal here)
    enable_internal_attacker_center = False
    enable_internal_attacker_server1 = False
    attack_types = ["deep_leakage"]

    # ---- Log config ----
    config = {
        "n_client": n_client,
        "n_server1": n_server1,
        "n_epoch": n_epoch,
        "batch_size": batch_size,
        "dataset_name": dataset_name,
        "dirichlet_alpha": dirichlet_alpha,
        "model_type": model_type,
        "compression_mode": compression_mode,
        "compress_first_round": compress_first_round,
        "use_fixed_batch": use_fixed_batch,
    }
    logger.log_config(config)

    # Assign clients to server1s
    clients_per_server1 = n_client // n_server1
    client_assignments = []
    for i in range(n_server1):
        start_idx = i * clients_per_server1
        end_idx = n_client if i == n_server1 - 1 else (i + 1) * clients_per_server1
        client_assignments.append(list(range(start_idx, end_idx)))

    logger.info("\nClient assignments to Server1:")
    for i, assignment in enumerate(client_assignments):
        logger.info(f"  Server1 {i}: clients {assignment}")

    # Center server uses global test set for evaluation
    full_train_loader = DataLoader(data_loader.train_dataset, batch_size=batch_size, shuffle=True)
    full_test_loader = DataLoader(data_loader.test_dataset, batch_size=batch_size, shuffle=False)

    logger.info("\nInitialize Center Server...")
    center_server = server(
        size=n_server1,
        data_loader=(full_train_loader, full_test_loader),
        device=device,
        encryption_key=ENCRYPTION_KEY,
        n_class=n_class,
        logger=logger,
        enable_attack=enable_internal_attacker_center,
        attack_types=attack_types,
        compression_mode=compression_mode,
        in_dim=in_dim,
        model_type=model_type,
    )

    logger.info("Initialize Intermediate Servers (Server1)...")
    server1s = []
    for i in range(n_server1):
        server1s.append(
            server1(
                rank=i,
                client_ranks=client_assignments[i],
                logger=logger,
                enable_attack=enable_internal_attacker_server1,
                attack_types=attack_types,
                n_class=n_class,
                device=device,
                data_loader=data_loader,
            )
        )

    logger.info("Initialize Clients...")
    clients = []
    for i in range(n_client):
        train_loader, test_loader = data_loader.get_loader(i, use_global_test=True)
        clients.append(
            client(
                rank=i,
                data_loader=(train_loader, test_loader),
                device=device,
                encryption_key=ENCRYPTION_KEY,
                n_class=n_class,
                logger=logger,
                compression_mode=compression_mode,
                in_dim=in_dim,
                model_type=model_type,
            )
        )

    # ===== Training rounds =====
    for epoch in range(1, n_epoch + 1):
        logger.info(f"\n================ Round {epoch}/{n_epoch} ================")

        # Whether to enable compression this round
        enable_compression = True
        enable_compression_all = True
        if (epoch == 1) and (not compress_first_round):
            enable_compression = False
            enable_compression_all = False

        # 1) clients compute grads
        for c in clients:
            c.run(enable_compression=enable_compression, enable_compression_all=enable_compression_all, use_fixed_batch=use_fixed_batch)

        # 2) server1 aggregate
        for s1 in server1s:
            s1.aggregate(epoch=epoch)

        # 3) center aggregate + update model
        center_server.aggregate(epoch=epoch)

    logger.info("\nTraining finished.")


if __name__ == "__main__":
    os.makedirs("./cache", exist_ok=True)
    os.makedirs("./log", exist_ok=True)
    federated_learning()
