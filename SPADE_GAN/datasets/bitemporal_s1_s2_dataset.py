"""Paired bitemporal S1/S2 dataset for optical synthesis.

Supported optical targets:
  * ``s2_tci``: 3-channel RGB PNG/TIFF, scaled from uint8 to [-1, 1]
  * ``s2``: xBD-S12 12-band TIFF, per-band percentile scaled to [-1, 1]

No modality fallback is allowed.  A full-S2 run must read ``train_s2`` and
``val_s2``; it must never silently consume TCI files.
"""

import random
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from utils.optical import S1_P1, S1_P99, S2_P1, S2_P99, optical_channels


class BitemporalS1S2Dataset(Dataset):
    """Pair pre-S1, pre-S2, post-S1 and post-S2 by xBD sample id."""

    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        target_size: Tuple[int, int] = (128, 128),
        optical_mode: str = "s2_tci",
        sar_normalization: str = "auto",
        augment: bool = True,
    ):
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.target_size = tuple(target_size)
        self.optical_mode = optical_mode
        self.num_optical_channels = optical_channels(optical_mode)
        if sar_normalization == "auto":
            # Preserve compatibility with already-trained TCI checkpoints while
            # matching the official xBD-S12 convention for new full-S2 runs.
            sar_normalization = "legacy" if optical_mode == "s2_tci" else "xbd_s12"
        if sar_normalization not in {"legacy", "xbd_s12"}:
            raise ValueError("sar_normalization must be 'auto', 'legacy', or 'xbd_s12'.")
        self.sar_normalization = sar_normalization
        self.augment = augment and split == "train"

        self.s2_dir = self.dataset_path / f"{split}_{optical_mode}"
        self.s1_dir = self.dataset_path / f"{split}_s1"
        missing = [str(p) for p in (self.s1_dir, self.s2_dir) if not p.is_dir()]
        if missing:
            raise FileNotFoundError(
                f"Missing required {split} folder(s): {missing}. "
                f"optical_mode={optical_mode!r} requires exactly '{split}_{optical_mode}'."
            )

        self.samples = self._index_quadruplets()
        if not self.samples:
            raise RuntimeError(
                f"No complete pre/post S1 + {optical_mode} quadruplets found in {self.dataset_path}."
            )

    @staticmethod
    def _first_existing(candidates: List[Path]):
        return next((path for path in candidates if path.exists()), None)

    def _index_quadruplets(self) -> List[Dict[str, Path]]:
        samples: List[Dict[str, Path]] = []
        extensions = ("*.tif", "*.tiff") if self.optical_mode == "s2" else ("*.png", "*.tif", "*.tiff")
        post_files: List[Path] = []
        for extension in extensions:
            post_files.extend(self.s2_dir.glob(f"*_post_disaster*{extension[1:]}"))

        # A set prevents duplicate indexing if unusual filenames match more than once.
        for post_s2_path in sorted(set(post_files)):
            uid = post_s2_path.name.split("_post_disaster")[0]
            pre_s2_path = self._first_existing([
                self.s2_dir / post_s2_path.name.replace("_post_disaster", "_pre_disaster"),
                self.s2_dir / f"{uid}_pre_disaster_{self.optical_mode}.tif",
                self.s2_dir / f"{uid}_pre_disaster_{self.optical_mode}.tiff",
                self.s2_dir / f"{uid}_pre_disaster_{self.optical_mode}.png",
            ])
            pre_s1_path = self._first_existing([
                self.s1_dir / f"{uid}_pre_disaster_s1.tif",
                self.s1_dir / f"{uid}_pre_disaster_s1.tiff",
                self.s1_dir / f"{uid}_pre_disaster_s1.png",
            ])
            post_s1_path = self._first_existing([
                self.s1_dir / f"{uid}_post_disaster_s1.tif",
                self.s1_dir / f"{uid}_post_disaster_s1.tiff",
                self.s1_dir / f"{uid}_post_disaster_s1.png",
            ])
            if pre_s2_path and pre_s1_path and post_s1_path:
                samples.append({
                    "uid": uid,
                    "pre_s1": pre_s1_path,
                    "pre_s2": pre_s2_path,
                    "post_s1": post_s1_path,
                    "post_s2": post_s2_path,
                })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _read_raster(path: Path) -> np.ndarray:
        if path.suffix.lower() in {".tif", ".tiff"}:
            import tifffile

            arr = tifffile.imread(str(path))
        else:
            arr = np.asarray(Image.open(path))
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[..., None]
        # Accept both CHW and HWC TIFFs.
        if arr.ndim == 3 and arr.shape[0] in (2, 3, 12) and arr.shape[-1] not in (2, 3, 12):
            arr = np.transpose(arr, (1, 2, 0))
        return arr

    def _resize(self, arr: np.ndarray) -> np.ndarray:
        # PIL expects (width, height), whereas tensor/array shapes are (height, width).
        out_h, out_w = self.target_size
        if arr.shape[:2] == (out_h, out_w):
            return arr
        resized = np.empty((out_h, out_w, arr.shape[-1]), dtype=np.float32)
        for channel in range(arr.shape[-1]):
            band = Image.fromarray(arr[..., channel], mode="F")
            resized[..., channel] = np.asarray(
                band.resize((out_w, out_h), resample=Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
        return resized

    def _load_optical(self, path: Path) -> np.ndarray:
        arr = self._resize(self._read_raster(path))
        if arr.shape[-1] != self.num_optical_channels:
            raise ValueError(
                f"{path} has {arr.shape[-1]} channels; optical_mode={self.optical_mode!r} "
                f"requires {self.num_optical_channels}."
            )
        if self.optical_mode == "s2_tci":
            return np.clip(arr / 127.5 - 1.0, -1.0, 1.0).astype(np.float32)

        low = S2_P1.reshape(1, 1, 12)
        high = S2_P99.reshape(1, 1, 12)
        scaled = np.clip((arr - low) / (high - low), 0.0, 1.0)
        return (scaled * 2.0 - 1.0).astype(np.float32)

    def _load_sar(self, path: Path) -> np.ndarray:
        arr = self._resize(self._read_raster(path))
        if arr.shape[-1] < 2:
            raise ValueError(f"{path} has {arr.shape[-1]} channel(s); S1 requires VV and VH.")
        arr = arr[..., :2]
        if self.sar_normalization == "legacy":
            low = np.asarray([-25.0, -25.0], dtype=np.float32).reshape(1, 1, 2)
            high = np.asarray([0.0, 0.0], dtype=np.float32).reshape(1, 1, 2)
        else:
            low = S1_P1.reshape(1, 1, 2)
            high = S1_P99.reshape(1, 1, 2)
        scaled = np.clip((arr - low) / (high - low), 0.0, 1.0)
        return (scaled * 2.0 - 1.0).astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        item = self.samples[idx]
        tensors = {
            "pre_s1": torch.from_numpy(self._load_sar(item["pre_s1"])).permute(2, 0, 1),
            "pre_s2": torch.from_numpy(self._load_optical(item["pre_s2"])).permute(2, 0, 1),
            "post_s1": torch.from_numpy(self._load_sar(item["post_s1"])).permute(2, 0, 1),
            "post_s2": torch.from_numpy(self._load_optical(item["post_s2"])).permute(2, 0, 1),
        }

        if self.augment:
            if random.random() > 0.5:
                tensors = {key: TF.hflip(value) for key, value in tensors.items()}
            if random.random() > 0.5:
                tensors = {key: TF.vflip(value) for key, value in tensors.items()}
            rot_k = random.choice((0, 1, 2, 3))
            if rot_k:
                tensors = {
                    key: torch.rot90(value, k=rot_k, dims=(1, 2))
                    for key, value in tensors.items()
                }

        return {"uid": item["uid"], **{key: value.float() for key, value in tensors.items()}}