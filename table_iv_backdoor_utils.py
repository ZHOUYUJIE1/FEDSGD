import math
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def normalized_white_value(
    mean: Sequence[float] = CIFAR10_MEAN,
    std: Sequence[float] = CIFAR10_STD,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    values = [(1.0 - m) / s for m, s in zip(mean, std)]
    return torch.tensor(values, dtype=dtype, device=device).view(-1, 1, 1)


def add_trigger_patch(
    images: torch.Tensor,
    trigger_size: int = 3,
    trigger_location: str = "bottom-right",
    trigger_value: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if trigger_size <= 0:
        return images

    squeeze = False
    if images.dim() == 3:
        images = images.unsqueeze(0)
        squeeze = True
    if images.dim() != 4:
        raise ValueError(f"Expected CHW or BCHW tensor, got shape={tuple(images.shape)}")

    out = images.clone()
    _, channels, height, width = out.shape
    size = min(trigger_size, height, width)
    if trigger_value is None:
        trigger_value = torch.ones(channels, 1, 1, dtype=out.dtype, device=out.device)
    else:
        trigger_value = trigger_value.to(device=out.device, dtype=out.dtype)
        if trigger_value.dim() == 1:
            trigger_value = trigger_value.view(-1, 1, 1)

    if trigger_location == "bottom-right":
        row_slice = slice(height - size, height)
        col_slice = slice(width - size, width)
    elif trigger_location == "top-left":
        row_slice = slice(0, size)
        col_slice = slice(0, size)
    elif trigger_location == "top-right":
        row_slice = slice(0, size)
        col_slice = slice(width - size, width)
    elif trigger_location == "bottom-left":
        row_slice = slice(height - size, height)
        col_slice = slice(0, size)
    else:
        raise ValueError(f"Unsupported trigger_location={trigger_location!r}")

    out[:, :, row_slice, col_slice] = trigger_value
    return out.squeeze(0) if squeeze else out


class ExactPoisonedDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        target_class: int,
        poison_ratio: float,
        seed: int,
        trigger_size: int = 3,
        trigger_location: str = "bottom-right",
    ) -> None:
        self.base_dataset = base_dataset
        self.target_class = int(target_class)
        self.poison_ratio = float(poison_ratio)
        self.trigger_size = int(trigger_size)
        self.trigger_location = trigger_location
        self.target_class_source = "assumption_from_cwt_reference_default_cifar10_target_9"

        total = len(base_dataset)
        self.poison_num = int(math.floor(self.poison_ratio * total))
        rng = np.random.RandomState(seed)
        if self.poison_num > 0:
            chosen = rng.choice(total, size=self.poison_num, replace=False)
            self.poison_indices = set(int(i) for i in chosen.tolist())
        else:
            self.poison_indices = set()

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        image, _label = self.base_dataset[index]
        if index in self.poison_indices:
            if not isinstance(image, torch.Tensor):
                image = torch.tensor(image)
            value = normalized_white_value(device=image.device, dtype=image.dtype)
            image = add_trigger_patch(
                image,
                trigger_size=self.trigger_size,
                trigger_location=self.trigger_location,
                trigger_value=value,
            )
            return image, self.target_class
        return image, _label


def make_triggered_tensor_dataset(
    clean_dataset: Dataset,
    target_class: int,
    trigger_size: int = 3,
    trigger_location: str = "bottom-right",
    batch_size: int = 256,
) -> TensorDataset:
    images = []
    labels = []
    loader = torch.utils.data.DataLoader(clean_dataset, batch_size=batch_size, shuffle=False)
    for batch_images, batch_labels in loader:
        value = normalized_white_value(device=batch_images.device, dtype=batch_images.dtype)
        images.append(
            add_trigger_patch(
                batch_images,
                trigger_size=trigger_size,
                trigger_location=trigger_location,
                trigger_value=value,
            )
        )
        labels.append(batch_labels)

    if not images:
        return TensorDataset(torch.empty(0), torch.empty(0, dtype=torch.long))
    return TensorDataset(torch.cat(images, dim=0), torch.cat(labels, dim=0))


@torch.no_grad()
def evaluate_accuracy(model: torch.nn.Module, loader, device: torch.device) -> float:
    model.eval()
    total = 0
    correct = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += int(preds.eq(labels).sum().item())
        total += int(labels.numel())
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_asr(model: torch.nn.Module, loader, target_class: int, device: torch.device) -> float:
    model.eval()
    total = 0
    hits = 0
    for images, _labels in loader:
        images = images.to(device)
        preds = model(images).argmax(dim=1)
        hits += int(preds.eq(int(target_class)).sum().item())
        total += int(preds.numel())
    return hits / max(total, 1)


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))
