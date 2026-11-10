import os
import random
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import tifffile as tiff
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader



class PairedS1S2Dataset(Dataset):
    """
    Paired dataset for Sentinel-1 (2-band SAR) -> post-disaster optical imagery.
    Pairs are matched strictly by unique tile identifier (UID).
    """

    def __init__(
        self,
        dataset_path: str,
        stage: str = "train",
        image_size: int = 128,
        s1_mean: Tuple[float, float] = (-12.2116, -19.7983),
        s1_std: Tuple[float, float] = (5.1599, 6.6504),
        flip: bool = True,
        target_mode: str = "s2",
        use_pre: bool = False,
    ):
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.stage = stage
        self.image_size = (image_size, image_size)
        self.s1_mean = torch.tensor(s1_mean, dtype=torch.float32).view(2, 1, 1)
        self.s1_std = torch.tensor(s1_std, dtype=torch.float32).view(2, 1, 1)
        self.flip = flip and (stage == "train")
        if target_mode not in {"s2", "tci"}:
            raise ValueError(f"target_mode must be 's2' or 'tci', got {target_mode!r}")
        self.target_mode = target_mode
        self.use_pre = use_pre

        stage_dirs = {
            "train": ("train_s1", "train_s2" if target_mode == "s2" else "train_s2_tci"),
            "val": ("val_s1", "val_s2" if target_mode == "s2" else "val_s2_tci"),
            "test": ("test_s1", "test_s2" if target_mode == "s2" else "test_s2_tci"),
        }

        s1_dir_name, s2_dir_name = stage_dirs.get(stage, ("train_s1", "train_s2"))
        self.s1_dir = self.dataset_path / s1_dir_name
        self.s2_dir = self.dataset_path / s2_dir_name

        # Fallback only for the legacy full-S2 layout.
        if not self.s1_dir.is_dir() or not self.s2_dir.is_dir():
            if target_mode == "s2" and (self.dataset_path / "s1").is_dir() and (self.dataset_path / "s2").is_dir():
                self.s1_dir = self.dataset_path / "s1"
                self.s2_dir = self.dataset_path / "s2"

        self.pairs: List[Tuple[Path, Path, Optional[Path], str]] = self._find_pairs()
        pre_info = " + pre_S2" if self.use_pre else ""
        print(f"[PairedS1S2Dataset] {stage} ({target_mode}{pre_info}): found {len(self.pairs)} pairs.")

    def _find_pairs(self) -> List[Tuple[Path, Path, Optional[Path], str]]:
        if not self.s1_dir.is_dir() or not self.s2_dir.is_dir():
            raise FileNotFoundError(
                f"Directories not found: S1={self.s1_dir}, S2={self.s2_dir}"
            )

        target_suffix = "_post_disaster_s2.tif" if self.target_mode == "s2" else "_post_disaster_s2_tci.tif"
        pre_suffix = "_pre_disaster_s2.tif" if self.target_mode == "s2" else "_pre_disaster_s2_tci.tif"

        known_s2_post = {}
        known_s2_pre = {}
        for s2_file in sorted(self.s2_dir.glob("*.tif")):
            name = s2_file.name
            if name.endswith(target_suffix):
                uid = name[:-len(target_suffix)]
                known_s2_post[uid] = s2_file
            elif name.endswith(pre_suffix):
                uid = name[:-len(pre_suffix)]
                known_s2_pre[uid] = s2_file

        pairs = []
        for s1_file in sorted(self.s1_dir.glob("*.tif")):
            name = s1_file.name
            uid = None
            if name.endswith("_post_disaster_s1.tif"):
                uid = name[:-len("_post_disaster_s1.tif")]

            if uid is not None and uid in known_s2_post:
                pre_path = known_s2_pre.get(uid, None)
                if self.use_pre and pre_path is None:
                    continue  # Require pre_s2 if use_pre is enabled
                pairs.append((s1_file, known_s2_post[uid], pre_path, uid))

        if len(pairs) == 0:
            raise RuntimeError(
                f"No matched pairs found in:\n  S1: {self.s1_dir}\n  S2: {self.s2_dir}"
            )
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        s1_path, s2_path, pre_path, uid = self.pairs[idx]

        # 1. Load S1 (2-channel SAR float32, in dB)
        s1 = tiff.imread(str(s1_path)).astype(np.float32)
        if s1.ndim == 2:
            s1 = np.expand_dims(s1, axis=-1)
        # (H, W, C) -> (C, H, W)
        s1 = torch.from_numpy(s1).permute(2, 0, 1).contiguous()

        # 2. Load the matched post-disaster optical target.
        s2 = tiff.imread(str(s2_path)).astype(np.float32)
        if s2.ndim == 2:
            s2 = np.expand_dims(s2, axis=-1)
        # (H, W, C) -> (C, H, W)
        s2 = torch.from_numpy(s2).permute(2, 0, 1).contiguous()

        # 3. If use_pre is enabled, load pre-disaster optical condition
        pre_s2 = None
        if self.use_pre and pre_path is not None:
            pre_s2 = tiff.imread(str(pre_path)).astype(np.float32)
            if pre_s2.ndim == 2:
                pre_s2 = np.expand_dims(pre_s2, axis=-1)
            pre_s2 = torch.from_numpy(pre_s2).permute(2, 0, 1).contiguous()

        # 4. Resize if spatial dimensions differ from image_size
        if s1.shape[-2:] != self.image_size:
            s1 = F.interpolate(
                s1.unsqueeze(0), size=self.image_size, mode="bilinear", align_corners=False
            ).squeeze(0)

        if s2.shape[-2:] != self.image_size:
            s2 = F.interpolate(
                s2.unsqueeze(0), size=self.image_size, mode="bilinear", align_corners=False
            ).squeeze(0)

        if pre_s2 is not None and pre_s2.shape[-2:] != self.image_size:
            pre_s2 = F.interpolate(
                pre_s2.unsqueeze(0), size=self.image_size, mode="bilinear", align_corners=False
            ).squeeze(0)

        # 5. Normalize S1: (dB - mean) / std, clamped to [-3, 3] / 3 -> [-1, 1]
        s1 = (s1 - self.s1_mean) / self.s1_std
        s1 = torch.clamp(s1, -3.0, 3.0) / 3.0

        # 6. Normalize target s2 and condition pre_s2 to [-1, 1]
        if self.target_mode == "tci":
            if s2.shape[0] != 3:
                raise ValueError(f"Expected 3-channel TCI, got {tuple(s2.shape)} from {s2_path}")
            s2 = torch.clamp(s2 / 127.5 - 1.0, -1.0, 1.0)
            if pre_s2 is not None:
                pre_s2 = torch.clamp(pre_s2 / 127.5 - 1.0, -1.0, 1.0)
        else:
            if s2.shape[0] != 12:
                raise ValueError(f"Expected 12-channel S2, got {tuple(s2.shape)} from {s2_path}")
            s2 = torch.clamp(s2 / 10000.0, 0.0, 1.0) * 2.0 - 1.0
            if pre_s2 is not None:
                pre_s2 = torch.clamp(pre_s2 / 10000.0, 0.0, 1.0) * 2.0 - 1.0

        # 7. Synchronized Data Augmentation for training
        if self.flip:
            if random.random() < 0.5:
                s1 = torch.flip(s1, dims=[2])  # horizontal flip
                s2 = torch.flip(s2, dims=[2])
                if pre_s2 is not None:
                    pre_s2 = torch.flip(pre_s2, dims=[2])
            if random.random() < 0.5:
                s1 = torch.flip(s1, dims=[1])  # vertical flip
                s2 = torch.flip(s2, dims=[1])
                if pre_s2 is not None:
                    pre_s2 = torch.flip(pre_s2, dims=[1])

        # 8. If use_pre is enabled, concatenate [post_S1, pre_S2] along channel dim
        # e.g., (2 + 3 = 5 channels)
        cond = torch.cat([s1, pre_s2], dim=0) if pre_s2 is not None else s1

        return cond, s2, uid


def create_dataloader(
    dataset_path: str,
    stage: str = "train",
    image_size: int = 128,
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: Optional[bool] = None,
    target_mode: str = "s2",
    use_pre: bool = False,
    s1_mean: Optional[Tuple[float, float]] = None,
    s1_std: Optional[Tuple[float, float]] = None,
) -> DataLoader:
    is_train = stage == "train"
    if shuffle is None:
        shuffle = is_train

    dataset_kwargs = {
        "dataset_path": dataset_path,
        "stage": stage,
        "image_size": image_size,
        "flip": is_train,
        "target_mode": target_mode,
        "use_pre": use_pre,
    }
    if s1_mean is not None:
        dataset_kwargs["s1_mean"] = tuple(s1_mean)
    if s1_std is not None:
        dataset_kwargs["s1_std"] = tuple(s1_std)

    dataset = PairedS1S2Dataset(**dataset_kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=is_train,
    )

