

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


IMPLEMENTATION_VERSION = "cc_function_order_v1"


@dataclass(frozen=True)
class CCVOFRWTConfig:
    levels: int = 3
    alpha_min: float = 0.15
    alpha_max: float = 1.25
    alpha_grid_size: int = 4
    energy_gain: float = 0.85
    gradient_gain: float = 0.65
    cross_channel_gain: float = 0.45
    eps: float = 1e-6

    def to_dict(self) -> dict:
        return {
            "implementation_version": IMPLEMENTATION_VERSION,
            **asdict(self),
        }


def _standardize_time(values: torch.Tensor, eps: float) -> torch.Tensor:
    mean = values.mean(dim=-1, keepdim=True)
    std = values.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
    return (values - mean) / std


class CrossChannelFunctionOrderTransform:


    statistics_per_band = 8

    def __init__(self, config: CCVOFRWTConfig | None = None):
        self.config = config or CCVOFRWTConfig()
        if self.config.levels < 1:
            raise ValueError("levels must be positive")
        if self.config.alpha_grid_size < 2:
            raise ValueError("alpha_grid_size must be at least 2")
        if self.config.alpha_min >= self.config.alpha_max:
            raise ValueError("alpha_min must be smaller than alpha_max")

    def feature_dim(self, n_channels: int) -> int:
        return (
            int(n_channels)
            * (self.config.levels + 1)
            * self.statistics_per_band
        )

    @staticmethod
    def _haar_bands(x: torch.Tensor, levels: int) -> list[torch.Tensor]:
        current = x
        details = []
        scale = float(2.0**-0.5)
        for _ in range(levels):
            if current.size(-1) % 2:
                current = torch.nn.functional.pad(current, (0, 1), mode="replicate")
            even = current[..., 0::2]
            odd = current[..., 1::2]
            details.append((even - odd) * scale)
            current = (even + odd) * scale
        return [current, *reversed(details)]

    def _order_function(self, band: torch.Tensor, band_index: int) -> torch.Tensor:
        cfg = self.config
        energy = _standardize_time(band.abs(), cfg.eps)
        gradient = torch.diff(band, dim=-1, prepend=band[..., :1]).abs()
        gradient = _standardize_time(gradient, cfg.eps)
        channel_center = band.mean(dim=1, keepdim=True)
        cross_channel = _standardize_time((band - channel_center).abs(), cfg.eps)

        band_fraction = band_index / max(cfg.levels, 1)
        base_alpha = cfg.alpha_min + (
            cfg.alpha_max - cfg.alpha_min
        ) * (0.25 + 0.50 * band_fraction)
        normalized = (base_alpha - cfg.alpha_min) / (
            cfg.alpha_max - cfg.alpha_min
        )
        base_logit = float(np.log(normalized / (1.0 - normalized)))
        logits = (
            base_logit
            + cfg.energy_gain * energy
            + cfg.gradient_gain * gradient
            + cfg.cross_channel_gain * cross_channel
        )
        return cfg.alpha_min + (
            cfg.alpha_max - cfg.alpha_min
        ) * torch.sigmoid(logits)

    def _fractional_bank(
        self,
        band: torch.Tensor,
        order: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.config
        length = band.size(-1)
        grid = torch.linspace(
            cfg.alpha_min,
            cfg.alpha_max,
            cfg.alpha_grid_size,
            dtype=band.dtype,
            device=band.device,
        )
        omega = 2.0 * torch.pi * torch.fft.fftfreq(
            length, dtype=band.dtype, device=band.device
        )
        abs_omega = omega.abs().clamp_min(cfg.eps)
        sign_omega = omega.sign()
        kernels = []
        for alpha in grid:
            kernel = abs_omega.pow(alpha) * torch.exp(
                0.5j * torch.pi * alpha * sign_omega
            )
            kernel = kernel.clone()
            kernel[0] = 0.0
            kernels.append(kernel)
        kernels = torch.stack(kernels, dim=0)

        spectrum = torch.fft.fft(band, dim=-1)
        bank = torch.fft.ifft(
            spectrum.unsqueeze(2) * kernels.view(1, 1, -1, length),
            dim=-1,
        ).real

        upper = torch.searchsorted(grid, order.contiguous()).clamp(
            1, cfg.alpha_grid_size - 1
        )
        lower = upper - 1
        lower_order = grid[lower]
        upper_order = grid[upper]
        weight = (order - lower_order) / (
            upper_order - lower_order
        ).clamp_min(cfg.eps)
        lower_value = bank.gather(2, lower.unsqueeze(2)).squeeze(2)
        upper_value = bank.gather(2, upper.unsqueeze(2)).squeeze(2)
        return lower_value + weight * (upper_value - lower_value)

    def transform_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Return descriptors for x shaped [batch, time=64, channels]."""
        if x.ndim != 3:
            raise ValueError(f"expected [batch,time,channels], got {tuple(x.shape)}")
        if x.size(1) < 2**self.config.levels:
            raise ValueError("sequence is too short for the configured wavelet levels")
        channels_first = x.transpose(1, 2).contiguous()
        feature_blocks = []
        for band_index, band in enumerate(
            self._haar_bands(channels_first, self.config.levels)
        ):
            order = self._order_function(band, band_index)
            transformed = self._fractional_bank(band, order)
            abs_value = transformed.abs()
            energy = transformed.square().mean(dim=-1)
            probability = abs_value / abs_value.sum(
                dim=-1, keepdim=True
            ).clamp_min(self.config.eps)
            entropy = -(probability * probability.clamp_min(
                self.config.eps
            ).log()).sum(dim=-1)
            stats = torch.stack(
                [
                    transformed.mean(dim=-1),
                    transformed.std(dim=-1, unbiased=False),
                    transformed.square().mean(dim=-1).sqrt(),
                    abs_value.amax(dim=-1),
                    torch.log1p(energy),
                    entropy,
                    order.mean(dim=-1),
                    order.std(dim=-1, unbiased=False),
                ],
                dim=-1,
            )
            feature_blocks.append(stats.flatten(start_dim=1))
        return torch.cat(feature_blocks, dim=1)

    @torch.inference_mode()
    def extract_numpy(
        self,
        samples: np.ndarray,
        batch_size: int = 1024,
        device: str | torch.device | None = None,
    ) -> np.ndarray:
        device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        outputs = []
        for start in range(0, len(samples), batch_size):
            batch = torch.as_tensor(
                np.asarray(samples[start : start + batch_size]),
                dtype=torch.float32,
                device=device,
            )
            outputs.append(self.transform_batch(batch).cpu().numpy())
        if not outputs:
            return np.empty((0, self.feature_dim(samples.shape[-1])), np.float32)
        return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)

    @torch.inference_mode()
    def extract_to_npy(
        self,
        samples: np.ndarray,
        output_path: str | Path,
        batch_size: int = 1024,
        device: str | torch.device | None = None,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(samples), self.feature_dim(samples.shape[-1])),
        )
        device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        for start in range(0, len(samples), batch_size):
            stop = min(start + batch_size, len(samples))
            batch = torch.as_tensor(
                np.asarray(samples[start:stop]),
                dtype=torch.float32,
                device=device,
            )
            output[start:stop] = self.transform_batch(batch).cpu().numpy()
        output.flush()
        del output
        return output_path


def summarize_features(features: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(features, dtype=np.float64)
    return {
        "samples": int(values.shape[0]),
        "feature_dim": int(values.shape[1]),
        "mean_abs": float(np.mean(np.abs(values))),
        "mean_std": float(np.mean(np.std(values, axis=0))),
        "max_abs": float(np.max(np.abs(values))),
    }
