import math
from typing import Dict, Optional, Tuple, List

import torch


class CATECompressor:
    """
    FL-CATE (CAT sparsification + error compensation) compressor.

    Paper idea:
      - At each round k (and per client i), pick sparsification level B by:
            B* = argmax_B  Efficiency_B(Δ)
        where Efficiency_B(Δ) = Information_B(Δ) / Cost_B(Δ)
              Information_B(Δ) = ||S_B(Δ)||^2 / ||Δ||^2
      - Use error compensation (a.k.a. error feedback):
            transmit S_B(Δ + e)
            update e <- (Δ + e) - S_B(Δ + e)

    We implement this per-tensor (per-layer) so you can drop it into your
    existing client loop that compresses each parameter's gradient tensor.
    References: Algorithm 1 (steps 8-13) and Eq.(2)(3) in the paper.
    """

    def __init__(
        self,
        *,
        value_bits: int = 32,
        packet_overhead_bits: int = 0,
        index_overhead_bits: int = 0,
        cost_model: str = "bits",
        candidate_ratios: Optional[List[float]] = None,
        min_B: int = 1,
        max_ratio: float = 1.0,
        device: Optional[torch.device] = None,
        logger=None,
    ) -> None:
        """
        Args:
            value_bits: bits used to transmit each value (32 for fp32; set to 8 etc if you later quantize)
            packet_overhead_bits: constant overhead per message (packet header etc.). For pure "bits transmitted"
                                 experiments you can keep it 0, or set something like 8*133 etc.
            index_overhead_bits: extra bits per index (e.g., run-length headers, masks). Default 0.
            cost_model: currently only "bits". (paper supports energy/latency too; you can plug your own.)
            candidate_ratios: candidate sparsity ratios for searching B. If None, use a reasonable default grid.
            min_B: minimal number of kept entries.
            max_ratio: upper bound ratio for kept entries (<=1.0).
            device: device for residual buffers (defaults to tensor's device at runtime).
            logger: optional logger with .info/.warning.
        """
        self.value_bits = int(value_bits)
        self.packet_overhead_bits = int(packet_overhead_bits)
        self.index_overhead_bits = int(index_overhead_bits)
        self.cost_model = str(cost_model)
        self.min_B = int(min_B)
        self.max_ratio = float(max_ratio)
        self.device = device
        self.logger = logger

        if candidate_ratios is None:
            # Default search grid: from very sparse to dense (log-ish, plus some common points)
            candidate_ratios = [
                1e-4, 2e-4, 5e-4,
                1e-3, 2e-3, 5e-3,
                1e-2, 2e-2, 5e-2,
                1e-1, 2e-1, 5e-1,
                1.0,
            ]
        self.candidate_ratios = [float(r) for r in candidate_ratios if r > 0]

        # Error feedback buffers: per-layer residual (flattened tensor stored in original shape)
        self._residuals: Dict[str, torch.Tensor] = {}
        self._last_B: Dict[str, int] = {}
        self._last_eff: Dict[str, float] = {}

    def reset_error_feedback(self) -> None:
        self._residuals.clear()
        self._last_B.clear()
        self._last_eff.clear()

    def get_last_selected_B(self, name: str) -> Optional[int]:
        return self._last_B.get(name)

    def get_last_efficiency(self, name: str) -> Optional[float]:
        return self._last_eff.get(name)

    # ----------------------- Core math from paper -----------------------

    def _information_ratio(self, v_flat: torch.Tensor, topk_abs2_prefix: torch.Tensor, B: int) -> float:
        # Information_B(v) = ||S_B(v)||^2 / ||v||^2
        total = float(torch.sum(v_flat * v_flat).item())
        if total <= 0.0 or not math.isfinite(total):
            return 0.0
        kept = float(topk_abs2_prefix[B - 1].item())
        return kept / total

    def _cost_bits(self, d: int, B: int) -> float:
        """
        Cost_B in "bits" model.
        You can customize this to match your communication stack.

        Minimal top-k encoding cost:
          - B values, each uses value_bits
          - B indices, each uses ceil(log2(d)) bits
          - optional per-index overhead (index_overhead_bits)
          - optional per-message overhead (packet_overhead_bits)
        """
        if B <= 0:
            return float("inf")
        index_bits = math.ceil(math.log2(max(d, 2)))
        return float(self.packet_overhead_bits + B * (self.value_bits + index_bits + self.index_overhead_bits))

    def _efficiency(self, d: int, info: float, B: int) -> float:
        if self.cost_model != "bits":
            # Hook: if you want energy/latency, implement your own and set cost_model accordingly.
            cost = self._cost_bits(d, B)
        else:
            cost = self._cost_bits(d, B)

        if cost <= 0.0 or not math.isfinite(cost):
            return -float("inf")
        return info / cost

    def _select_B(self, v_flat: torch.Tensor, name: str) -> int:
        """
        Exhaustive 1-D search: B* = argmax_B Efficiency_B(v) (paper notes it is fast).
        See: B_k^i = argmax_{B in [1,d]} Efficiency_B(Δ_k^i) (paper).
        """
        d = v_flat.numel()
        max_B = max(self.min_B, int(self.max_ratio * d))
        max_B = min(max_B, d)

        # If nearly all zeros, just return min_B
        if torch.count_nonzero(v_flat).item() == 0:
            self._last_B[name] = self.min_B
            self._last_eff[name] = 0.0
            return self.min_B

        # Precompute sorted squared magnitudes and prefix sums for fast Info_B evaluation
        abs2 = (v_flat * v_flat).abs()  # v^2
        sorted_abs2, _ = torch.sort(abs2, descending=True)
        prefix = torch.cumsum(sorted_abs2, dim=0)

        best_B = self.min_B
        best_eff = -float("inf")

        # Candidate B values from ratio grid (clamped), plus endpoints
        candidates = set()
        candidates.add(self.min_B)
        candidates.add(max_B)
        for r in self.candidate_ratios:
            B = int(round(r * d))
            B = max(self.min_B, min(B, max_B))
            candidates.add(B)

        # Optional: densify candidates a bit around log-scale for smoother behavior
        # (kept small to avoid overhead)
        for B in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            if self.min_B <= B <= max_B:
                candidates.add(B)

        for B in sorted(candidates):
            info = self._information_ratio(v_flat, prefix, B)
            eff = self._efficiency(d, info, B)
            if eff > best_eff:
                best_eff = eff
                best_B = B

        self._last_B[name] = int(best_B)
        self._last_eff[name] = float(best_eff) if math.isfinite(best_eff) else 0.0
        return int(best_B)

    # ----------------------- Public API (match your client loop) -----------------------

    @torch.no_grad()
    def compress_tensor(
        self,
        tensor: torch.Tensor,
        *,
        name: str = "unnamed",
        global_gradient: Optional[torch.Tensor] = None,  # kept for signature-compatibility; not used here
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            values: (B,) tensor of selected entries (dense values)
            indices: (B,) int64 tensor of flattened indices
        """
        if tensor is None:
            raise ValueError("compress_tensor got None tensor")

        # Initialize / move residual buffer
        if name not in self._residuals or self._residuals[name].shape != tensor.shape:
            self._residuals[name] = torch.zeros_like(tensor, device=tensor.device)
        else:
            # keep residual on the same device as tensor
            if self._residuals[name].device != tensor.device:
                self._residuals[name] = self._residuals[name].to(tensor.device)

        v = tensor.detach() + self._residuals[name]  # Δ + e  (paper Algorithm 1 line 10)
        v_flat = v.reshape(-1)

        d = v_flat.numel()
        B = self._select_B(v_flat, name=name)

        if B >= d:
            # no sparsification
            indices = torch.arange(d, device=v_flat.device, dtype=torch.long)
            values = v_flat.clone()
            self._residuals[name].zero_()
            return values, indices

        # Top-k by magnitude
        _, topk_idx = torch.topk(v_flat.abs(), k=B, largest=True, sorted=False)
        values = v_flat[topk_idx].clone()
        indices = topk_idx.to(torch.long)

        # Error compensation update: e <- (Δ + e) - S_B(Δ + e)  (paper Algorithm 1 line 11)
        recon_flat = torch.zeros_like(v_flat)
        recon_flat[indices] = values
        residual_flat = v_flat - recon_flat
        self._residuals[name] = residual_flat.reshape_as(tensor).detach()

        return values, indices

    @torch.no_grad()
    def decompress_tensor(self, values: torch.Tensor, indices: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        """Dense reconstruction of the sparse tensor (zeros everywhere else)."""
        flat = torch.zeros(int(torch.prod(torch.tensor(shape)).item()), device=values.device, dtype=values.dtype)
        if values.numel() > 0:
            flat[indices.to(torch.long)] = values
        return flat.reshape(shape)
