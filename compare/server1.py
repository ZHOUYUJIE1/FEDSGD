import glob
import os
from typing import Dict, List, Optional

import torch

# Optional decrypt support for your TopkCompressor encryption stage
try:
    from topk import TopkCompressor
except Exception:
    TopkCompressor = None


class server1(object):
    """
    Intermediate server (Server1):
      - reads per-client gradient payloads from ./cache/grads_{client_rank}.pkl
      - (optionally) decrypts if payload['encrypted'] is True and an encryption key is available
      - aggregates (weighted average by n_samples) across its assigned clients
      - writes aggregated payload to ./cache/grads_server1_{rank}.pkl so center server can pick it up

    This implementation is compatible with:
      - compression_mode = 'none' / 'fixed' / 'adaptive' : client payloads follow your TopkCompressor pipeline
      - compression_mode = 'cate'                         : client payloads are dense (zero-filled), encrypted=False
    """

    def __init__(
        self,
        rank: int,
        client_ranks: List[int],
        logger=None,
        enable_attack: bool = False,
        attack_types: Optional[List[str]] = None,
        n_class: int = 10,
        device: Optional[torch.device] = None,
        data_loader=None,  # kept for signature compatibility; not required for aggregation
        encryption_key: Optional[str] = None,
    ):
        self.rank = int(rank)
        self.client_ranks = list(client_ranks)
        self.logger = logger
        self.enable_attack = bool(enable_attack)
        self.attack_types = attack_types or []
        self.n_class = int(n_class)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Try to discover encryption key if not provided (so you don't have to change main.py)
        self.encryption_key = encryption_key or os.environ.get("FEDSGD_ENCRYPTION_KEY", None)
        if self.encryption_key is None:
            # optional: read from cache file if you store it there
            for fp in ["./cache/encryption_key.txt", "./cache/encryption_key.pkl"]:
                if os.path.exists(fp):
                    try:
                        if fp.endswith(".txt"):
                            self.encryption_key = open(fp, "r", encoding="utf-8").read().strip()
                        else:
                            self.encryption_key = torch.load(fp)
                        break
                    except Exception:
                        pass

        # Decryptor (only if TopkCompressor exists + key is available)
        self._decryptor = None
        if TopkCompressor is not None and self.encryption_key is not None:
            try:
                self._decryptor = TopkCompressor(
                    compress_ratio=1.0,
                    min_sparsity_ratio=0.0,
                    max_sparsity_ratio=1.0,
                    encryption_key=self.encryption_key,
                    logger=self.logger,
                    compression_mode="none",
                )
            except Exception as e:
                self._warn(f"[Server1 {self.rank}] TopkCompressor init failed; cannot decrypt encrypted grads. err={e}")

    # ---------------- logging helpers ----------------
    def _info(self, msg: str):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def _warn(self, msg: str):
        if self.logger:
            self.logger.warning(msg)
        else:
            print("WARNING:", msg)

    # ---------------- IO ----------------
    def _client_grad_path(self, client_rank: int) -> str:
        return f"./cache/grads_{client_rank}.pkl"

    # ---------------- core aggregation ----------------
    def aggregate(self, epoch: int = 0):
        """
        Aggregate gradients from assigned clients.
        Output file: ./cache/grads_server1_{rank}.pkl
        """
        os.makedirs("./cache", exist_ok=True)

        payloads = []
        missing = 0
        for cr in self.client_ranks:
            fp = self._client_grad_path(cr)
            if not os.path.exists(fp):
                missing += 1
                continue
            try:
                payloads.append(torch.load(fp, map_location=self.device))
            except Exception as e:
                self._warn(f"[Server1 {self.rank}] Failed to load {fp}: {e}")

        if len(payloads) == 0:
            self._warn(f"[Server1 {self.rank}] No client payloads found (missing={missing}). Skip.")
            return

        # Decrypt if needed (only for your TopkCompressor encryption stage)
        for p in payloads:
            if not isinstance(p, dict):
                continue
            if bool(p.get("encrypted", False)):
                if self._decryptor is None:
                    self._warn(f"[Server1 {self.rank}] Encrypted grads present but no decryptor/key; cannot aggregate reliably.")
                    continue
                try:
                    named = p.get("named_grads", {})
                    if hasattr(self._decryptor, "decrypt_with_encryption"):
                        p["named_grads"] = self._decryptor.decrypt_with_encryption(named)
                        p["encrypted"] = False
                    elif hasattr(self._decryptor, "decompress_with_encryption"):
                        p["named_grads"] = self._decryptor.decompress_with_encryption(named)
                        p["encrypted"] = False
                    else:
                        self._warn(f"[Server1 {self.rank}] Decryptor missing decrypt_with_encryption/decompress_with_encryption.")
                except Exception as e:
                    self._warn(f"[Server1 {self.rank}] Decrypt failed: {e}")

        # Weighted average by n_samples
        agg: Dict[str, torch.Tensor] = {}
        total_w = 0.0

        for p in payloads:
            if not isinstance(p, dict):
                continue
            named: Dict[str, torch.Tensor] = p.get("named_grads", {})
            w = float(p.get("n_samples", 1.0))
            if w <= 0:
                w = 1.0
            total_w += w

            for name, g in named.items():
                if g is None:
                    continue
                gt = g.to(self.device)
                if name not in agg:
                    agg[name] = gt * w
                else:
                    agg[name] += gt * w

        if total_w <= 0:
            total_w = 1.0
        for name in list(agg.keys()):
            agg[name] = agg[name] / total_w

        out_payload = {
            "n_samples": total_w,  # for center weighting across server1s
            "named_grads": agg,
            "encrypted": False,  # aggregated grads are now plaintext tensors
            "compression_mode": payloads[0].get("compression_mode", "unknown"),
            "source": "server1",
            "server1_rank": self.rank,
            "epoch": epoch,
        }

        out_fp = f"./cache/grads_server1_{self.rank}.pkl"
        torch.save(out_payload, out_fp)
        self._info(f"[Server1 {self.rank}] Round {epoch}: saved aggregated grads -> {out_fp} (clients={len(payloads)}/{len(self.client_ranks)})")

        # Optional: hook for internal attacker on server1 aggregated grads
        if self.enable_attack:
            self._warn(f"[Server1 {self.rank}] enable_attack=True, but attacker hook is not wired in this replacement file.")
