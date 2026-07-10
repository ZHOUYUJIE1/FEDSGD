import os
import argparse
import torch

from model.lenet import lenet5, resnet18, cifar10cnn_v2
from model.attacker_modify_fixed import InternalAttacker  # 你实际在用的攻击代码文件
from model.data import loader  # 仅用于提供 data_loader 给 attacker（一些攻击里会用到）


def build_model(model_type: str, n_class: int, in_dim: int, device: torch.device):
    if model_type.lower() == "resnet18":
        model = resnet18(n_class=n_class, in_dim=in_dim).to(device)
    elif model_type.lower() == "cifar10cnn_v2":
        model = cifar10cnn_v2(n_class=n_class, in_dim=in_dim).to(device)
    else:
        model = lenet5(n_class=n_class, in_dim=in_dim).to(device)
    return model


def load_model_state_if_exists(model, state_path: str, device: torch.device):
    """尽量加载与梯度一致的模型参数，否则攻击会差很多"""
    if os.path.exists(state_path):
        sd = torch.load(state_path, map_location=device)
        model.load_state_dict(sd, strict=True)
        print(f"[OK] Loaded model state: {state_path}")
        return True
    else:
        print(f"[WARN] Model state not found: {state_path}. Using random init (attack quality may degrade).")
        return False


def load_grad_pkl(path: str, device: torch.device):
    obj = torch.load(path, map_location=device)
    # 兼容你保存结构：{"n_samples":..., "named_grads":..., "compression_mode":..., "encrypted":...}
    if isinstance(obj, dict) and "named_grads" in obj:
        grads = obj["named_grads"]
        meta = obj
    else:
        grads = obj
        meta = {}
    # 确保 tensor 在 device
    for k, v in list(grads.items()):
        if torch.is_tensor(v):
            grads[k] = v.to(device)
    return grads, meta


def infer_dataset_params(dataset: str):
    ds = dataset.lower()
    if ds == "mnist":
        return 1, (1, 28, 28)
    elif ds in ("cifar10", "cifar"):
        return 3, (3, 32, 32)
    else:
        raise ValueError(f"Unknown dataset={dataset}. Use mnist or cifar10.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grads_dir", type=str, default="./FedSGD/cache", help="目录中包含 grads_{rank}.pkl")
    parser.add_argument("--ranks", type=str, default="0", help="要攻击的客户端rank列表，例如 '0,1,2'")
    parser.add_argument("--epoch", type=int, default=1, help="用于命名保存文件的epoch编号（不影响攻击本身）")
    parser.add_argument("--attack_types", type=str, default="deep_leakage",
                        help="用逗号分隔，例如 'deep_leakage' 或 'gradient_inversion,deep_leakage'")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument("--model_type", type=str, default="lenet5", choices=["lenet5", "resnet18", "cifar10cnn_v2"])
    parser.add_argument("--n_class", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1, help="攻击重建的batch_size（必须与梯度生成时一致）")
    parser.add_argument("--num_iterations", type=int, default=3000, help="DLG迭代步数（如果你的攻击代码支持传参）")
    parser.add_argument("--device", type=str, default="cuda", help="cuda / cpu")
    parser.add_argument("--model_state_path", type=str, default="./FedSGD/cache/global_model_state.pkl",
                        help="尽量加载与梯度一致的模型参数")
    parser.add_argument("--save_visualization_epochs", type=str, default="1",
                        help="保存可视化的epoch列表，例如 '1,2'")
    parser.add_argument("--save_iteration_steps", type=str, default="1,50,200,1000",
                        help="保存中间重建的迭代步，例如 '1,50,200,1000'")
    args = parser.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"[INFO] device={device}")

    in_dim, (C, H, W) = infer_dataset_params(args.dataset)
    n_class = args.n_class

    # 1) 构建并加载模型（强烈建议加载 global_model_state.pkl）
    model = build_model(args.model_type, n_class=n_class, in_dim=in_dim, device=device)
    load_model_state_if_exists(model, args.model_state_path, device=device)

    # 2) 准备 data_loader（有些攻击实现会用到；即便不用，也传进去保持接口一致）
    #    注意：这里不跑训练，只是初始化 loader 供 attacker 使用。
    #    你可以根据需要改 batch_size / alpha / n_clients；对攻击本身一般不关键。
    try:
        data_loader = loader(args.dataset, batch_size=args.batch_size, n_clients=1, alpha=0.5, seed=42, use_augmentation=False)
        dl_tuple = data_loader.get_data_loader()
    except Exception as e:
        print(f"[WARN] Could not init data loader (not critical for attack in many cases): {e}")
        dl_tuple = None

    # 3) 解析保存配置
    attack_types = [s.strip() for s in args.attack_types.split(",") if s.strip()]
    save_visualization_epochs = [int(x) for x in args.save_visualization_epochs.split(",") if x.strip()]
    save_iteration_steps = [int(x) for x in args.save_iteration_steps.split(",") if x.strip()]

    # 4) 初始化攻击者
    attacker = InternalAttacker(
        attacker_type="external_attacker",
        attacker_id="external_attacker_from_grads",
        logger=None,                 # 你如果有 logger 可以替换
        model=model,
        n_class=n_class,
        device=device,
        data_loader=dl_tuple,
        save_visualization_epochs=save_visualization_epochs,
        save_iteration_steps=save_iteration_steps,
    )

    # 给 attacker 一个 input_shape（当 original_data=None 时用于初始化 dummy）
    # 这里是 (B,C,H,W)
    attacker.input_shape = (args.batch_size, C, H, W)
    attacker.dataset_name = args.dataset

    # 5) 逐个rank读取梯度并攻击
    ranks = [int(x) for x in args.ranks.split(",") if x.strip()]
    for r in ranks:
        grad_path = os.path.join(args.grads_dir, f"grads_{r}.pkl")
        if not os.path.exists(grad_path):
            print(f"[SKIP] Not found: {grad_path}")
            continue

        print(f"\n[ATTACK] rank={r} grad_path={grad_path}")
        gradients, meta = load_grad_pkl(grad_path, device=device)

        # 可选：打印一些meta
        if isinstance(meta, dict) and len(meta) > 0:
            cm = meta.get("compression_mode", "unknown")
            enc = meta.get("encrypted", "unknown")
            print(f"[META] compression_mode={cm}, encrypted={enc}")

        # 注意：GI 需要 original_labels，否则会跳过（你代码里就是这样）
        # 这里我们不提供 original_data/original_labels，因此只推荐跑 deep_leakage。
        results = attacker.attack_on_gradients(
            gradients=gradients,
            epoch=args.epoch,
            client_rank=r,
            attack_types=attack_types
        )

        print(f"[DONE] rank={r} results_keys={list(results.keys()) if isinstance(results, dict) else type(results)}")


if __name__ == "__main__":
    main()
