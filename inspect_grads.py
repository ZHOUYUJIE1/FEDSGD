import torch

path = "./cache/accuracy.pkl"   # 如果你要看别的 client，改成 grads_0.pkl 等

grads = torch.load(path, map_location="cpu")

print("====== keys ======")
print(grads.keys())

print("\n====== meta info ======")
for k in [
    "round",
    "client_rank",
    "fixed_batch",
    "compression_mode",
    "encrypted",
    "enable_compression",
    "enable_compression_all",
]:
    if k in grads:
        print(f"{k}: {grads[k]}")

print("\n====== gradient tensors ======")
named_grads = grads.get("named_grads", {})
for name, g in named_grads.items():
    if torch.is_tensor(g):
        nz = (g != 0).float().mean().item()
        print(
            f"{name:30s} shape={tuple(g.shape)}, "
            f"mean={g.mean().item():+.3e}, "
            f"std={g.std().item():.3e}, "
            f"nonzero_ratio={nz:.4f}"
        )
    else:
        print(f"{name:30s} type={type(g)}")
