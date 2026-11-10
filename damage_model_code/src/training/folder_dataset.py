"""Dataset loader for pre-partitioned xBD dataset directories."""

import json
import warnings
from pathlib import Path

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
import torch
import torch.nn.functional as F
from torch.utils import data as tdata

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

from src.constants import S1_BANDS, S2_BANDS
from src.training.utils import apply_buffer_around_buildings, downsample_categorical_mask

# Official Sentinel-1 and Sentinel-2 percentile statistics for normalization
DEFAULT_STATS = {
    "s1": {
        "1st": [-23.85894, -38.00315],
        "99th": [-0.83129, -8.70669],
    },
    "s2": {
        "1st": [1066.0, 1060.0, 1150.0, 1083.0, 1095.0, 1047.0, 1059.0, 1047.0, 1045.0, 1040.0, 1055.0, 1049.0],
        "99th": [2385.0, 2816.0, 3321.0, 4243.0, 4613.0, 4859.0, 5339.0, 5703.0, 5705.0, 5576.0, 6007.0, 5419.0],
    },
}


def plot_mask(mask: np.ndarray, ax: mpl.axes.Axes, show_legend: bool = False):
    """Plot categorical mask without requiring external raster libraries."""
    from src.constants import COLORS
    cmap = mpl.colors.ListedColormap([COLORS.get(i, "#000000") for i in range(3)])
    ax.imshow(mask, cmap=cmap, vmin=0, vmax=2)


class FolderXBDS12Dataset(tdata.Dataset):
    """Dataset for ``{split}_{modality}`` and ``{split}_masks`` folders."""

    MODALITIES = ["s1", "s2", "s2_tci"]
    IMAGE_SUFFIXES = (".tif", ".tiff", ".png")
    MASK_SUFFIXES = (".tif", ".tiff", ".png")
    CLASS_COUNT_CACHE_VERSION = 1

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        modalities: list[str] | None = None,
        task: str = "multiclass",
        transforms=None,
        pixels_buffer_around_buildings: int = 3,
        downsample_factor: int = 8,
        s2_bands: str | list = "all",
        s1_bands: str | list = "all",
        verbose: bool = True,
        s2_tci_dir: str | Path | None = None,
        post_s2_tci_dir: str | Path | None = None,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset root directory does not exist: {self.data_dir}")

        # Split alias: accept 'valid' or 'val'
        self.split_raw = split
        self.split = "val" if split in ["valid", "val"] else split
        self.task = task
        self.transforms = transforms
        self.pixels_buffer_around_buildings = pixels_buffer_around_buildings
        self.downsample_factor = downsample_factor
        self.verbose = verbose

        # Modalities
        modalities = modalities or ["s1", "s2_tci"]
        if isinstance(modalities, str):
            modalities = [modalities]
        invalid_modalities = sorted(set(modalities) - set(self.MODALITIES))
        if invalid_modalities:
            raise ValueError(f"Unsupported modalities {invalid_modalities}; choose from {self.MODALITIES}")
        if task not in ["multiclass", "localization"]:
            raise ValueError("task must be either 'multiclass' or 'localization'")
        self.modalities = sorted(modalities)

        # Bands configuration
        if s2_bands == "all":
            self.s2_bands = S2_BANDS
        elif s2_bands == "rgb":
            self.s2_bands = ["B4", "B3", "B2"]
        else:
            self.s2_bands = [s2_bands] if isinstance(s2_bands, str) else s2_bands

        if s1_bands == "all":
            self.s1_bands = S1_BANDS
        else:
            self.s1_bands = [s1_bands] if isinstance(s1_bands, str) else s1_bands

        self.s2_bands_ids = [S2_BANDS.index(b) for b in self.s2_bands]
        self.s1_bands_ids = [S1_BANDS.index(b) for b in self.s1_bands]

        self.use_s1 = "s1" in self.modalities
        self.use_s2 = "s2" in self.modalities
        self.use_s2_tci = "s2_tci" in self.modalities
        self.periods = ["pre", "post"]
        self.s2_tci_dir = Path(s2_tci_dir) if s2_tci_dir else None
        self.post_s2_tci_dir = Path(post_s2_tci_dir) if post_s2_tci_dir else None

        # Setup normalization parameters
        self._setup_normalization()

        # Locate folders and collect samples
        self._locate_folders()
        self.samples = self._collect_samples()

        if not self.samples:
            raise ValueError(
                f"No paired samples found for split '{self.split}' and modalities {self.modalities}. "
                "Expected masks such as <uid>_mask.tif/.png and images such as "
                "<uid>_pre_disaster_<modality>.tif/.png / <uid>_post_disaster_<modality>.tif/.png."
            )

        if self.verbose:
            print(f"[{self.split_raw.upper()}] Loaded {len(self.samples)} valid samples from {self.data_dir}")

    def _setup_normalization(self):
        """Load normalization stats from JSON or fall back to precomputed defaults."""
        stats = DEFAULT_STATS
        norm_fp = self.data_dir / "normalization.json"
        if norm_fp.exists():
            try:
                with open(norm_fp) as f:
                    stats = json.load(f)
            except Exception as e:
                if self.verbose:
                    print(f"Could not read {norm_fp} ({e}), using default stats.")

        self.norm_params = {}
        if self.use_s1:
            s1_min = torch.tensor(stats["s1"]["1st"], dtype=torch.float32)
            s1_max = torch.tensor(stats["s1"]["99th"], dtype=torch.float32)
            self.norm_params["s1_min"] = s1_min[self.s1_bands_ids].view(-1, 1, 1)
            self.norm_params["s1_max"] = s1_max[self.s1_bands_ids].view(-1, 1, 1)

        if self.use_s2:
            s2_min = torch.tensor(stats["s2"]["1st"], dtype=torch.float32)
            s2_max = torch.tensor(stats["s2"]["99th"], dtype=torch.float32)
            self.norm_params["s2_min"] = s2_min[self.s2_bands_ids].view(-1, 1, 1)
            self.norm_params["s2_max"] = s2_max[self.s2_bands_ids].view(-1, 1, 1)

    def _locate_folders(self):
        """Identify modality and mask directories for this split."""
        split_prefix = self.split  # 'train', 'val', or 'test'
        alt_prefix = "valid" if self.split == "val" else None

        self.modality_dirs = {}
        for mod in self.modalities:
            if mod == "s2_tci" and self.s2_tci_dir is not None:
                if not self.s2_tci_dir.is_dir():
                    raise FileNotFoundError(f"Provided s2_tci directory does not exist: {self.s2_tci_dir}")
                self.modality_dirs["s2_tci"] = self.s2_tci_dir
                continue

            # Try {split}_{mod} then {alt_split}_{mod}
            candidates = [self.data_dir / f"{split_prefix}_{mod}"]
            if alt_prefix:
                candidates.append(self.data_dir / f"{alt_prefix}_{mod}")
            # Also try flat mod directory if available
            candidates.append(self.data_dir / mod)

            if mod == "s2_tci":
                candidates.extend([
                    self.data_dir / f"{split_prefix}_{mod}_png",
                    self.data_dir / f"{split_prefix}_s2_png",
                    self.data_dir / f"{split_prefix}_s2",
                    self.data_dir / f"{split_prefix}_synthetic_optical",
                    self.data_dir / f"{split_prefix}_synthetic",
                    self.data_dir / f"{split_prefix}_{mod}_real",
                    self.data_dir / f"{mod}_png",
                    self.data_dir / f"{mod}_preprocessed_png",
                    self.data_dir / "s2_png",
                    self.data_dir / "test_synthetic_optical",
                    self.data_dir / "synthetic_optical",
                ])
                if alt_prefix:
                    candidates.extend([
                        self.data_dir / f"{alt_prefix}_{mod}_png",
                        self.data_dir / f"{alt_prefix}_synthetic_optical",
                    ])
            elif mod == "s2":
                candidates.extend([
                    self.data_dir / f"{split_prefix}_s2_png",
                    self.data_dir / f"{split_prefix}_s2_tci",
                    self.data_dir / f"{split_prefix}_s2_tci_png",
                    self.data_dir / "s2_png",
                ])
            else:
                candidates.extend([
                    self.data_dir / f"{split_prefix}_{mod}_png",
                    self.data_dir / f"{mod}_png",
                ])

            found = None
            for p in candidates:
                if p.is_dir():
                    found = p
                    break
            if not found:
                raise FileNotFoundError(f"Could not locate folder for modality '{mod}' in {self.data_dir} (tried {candidates})")
            self.modality_dirs[mod] = found

        # Find mask folder
        mask_candidates = [
            self.data_dir / f"{split_prefix}_tif_masks",
            self.data_dir / f"{split_prefix}_s2_masks",
            self.data_dir / f"{split_prefix}_s2_tci_masks",
            self.data_dir / f"{split_prefix}_masks",
            self.data_dir / f"{split_prefix}_mask",
        ]
        if alt_prefix:
            mask_candidates.extend([
                self.data_dir / f"{alt_prefix}_tif_masks",
                self.data_dir / f"{alt_prefix}_s2_masks",
                self.data_dir / f"{alt_prefix}_s2_tci_masks",
                self.data_dir / f"{alt_prefix}_masks",
            ])
        mask_candidates.append(self.data_dir / "masks")

        self.mask_dir = None
        for p in mask_candidates:
            if p.is_dir():
                self.mask_dir = p
                break

        if self.mask_dir is None:
            # Fallback search for any directory matching *mask*
            for p in self.data_dir.glob(f"*{split_prefix}*mask*"):
                if p.is_dir():
                    self.mask_dir = p
                    break
            if self.mask_dir is None:
                raise FileNotFoundError(f"Could not find mask folder for split '{split_prefix}' in {self.data_dir}")

    def _collect_samples(self) -> list[dict]:
        """Scan directory and collect all paired samples."""
        samples = []
        mask_files = sorted(fp for fp in self.mask_dir.iterdir() if fp.is_file() and fp.suffix.lower() in self.MASK_SUFFIXES)

        for mask_fp in mask_files:
            uid = self._uid_from_mask(mask_fp)
            disaster = uid.split("_")[0]

            # Verify that all requested modalities have pre and post images
            valid = True
            mod_files = {}
            for mod in self.modalities:
                dir_m = self.modality_dirs[mod]
                pre_fp = self._find_image(dir_m, uid, "pre", mod)
                post_dir = self.post_s2_tci_dir if (mod == "s2_tci" and self.post_s2_tci_dir) else dir_m
                post_fp = self._find_image(post_dir, uid, "post", mod)

                if pre_fp is None or post_fp is None:
                    valid = False
                    break
                mod_files[f"{mod}_pre"] = pre_fp
                mod_files[f"{mod}_post"] = post_fp

            if valid:
                samples.append({
                    "uid": uid,
                    "disaster": disaster,
                    "mask_fp": mask_fp,
                    **mod_files,
                })

        return samples

    @staticmethod
    def _uid_from_mask(mask_fp: Path) -> str:
        """Return the sample UID from common xBD mask filename styles."""
        stem = mask_fp.stem
        for suffix in ["_mask", "_post_disaster_s2_tci", "_post_disaster_s2", "_post_disaster"]:
            if stem.endswith(suffix):
                return stem[: -len(suffix)]
        return stem

    def _find_image(self, folder: Path, uid: str, period: str, modality: str) -> Path | None:
        """Find an image while preferring the canonical xBD-S12 filename."""
        stems = [
            f"{uid}_{period}_disaster_{modality}",
            f"{uid}_{period}_{modality}",
            f"{uid}_{period}_disaster",
            f"{uid}_{period}",
        ]
        if modality == "s2_tci":
            stems.extend([
                f"{uid}_{period}_disaster_s2",
                f"{uid}_{period}_s2",
                f"{uid}_{period}_disaster_synthetic_optical",
                f"{uid}_{period}_synthetic_optical",
                f"{uid}_{period}_disaster_synthetic",
                f"{uid}_{period}_synthetic",
            ])
            if period == "post":
                stems.extend([
                    f"{uid}_post_disaster_synthetic_optical",
                    f"{uid}_synthetic_optical",
                    f"{uid}_synthetic",
                    f"{uid}",
                ])

        for stem in stems:
            for suffix in self.IMAGE_SUFFIXES:
                candidate = folder / f"{stem}{suffix}"
                if candidate.exists():
                    return candidate

        # Accommodate harmless extra filename tokens, but never guess if the
        # result is ambiguous.
        matches = sorted(
            fp
            for fp in folder.glob(f"{uid}_{period}*")
            if fp.is_file() and fp.suffix.lower() in self.IMAGE_SUFFIXES
        )
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            for suffix in self.IMAGE_SUFFIXES:
                for match in matches:
                    if match.suffix.lower() == suffix:
                        return match
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def get_in_channels(self) -> int:
        n_channels = 0
        if self.use_s2:
            n_channels += len(self.s2_bands)
        if self.use_s2_tci:
            n_channels += 3
        if self.use_s1:
            n_channels += len(self.s1_bands)
        return n_channels

    def get_out_channels(self) -> int:
        return 1 if self.task == "localization" else 3

    def _read_raster(self, fp: Path) -> torch.Tensor:
        """Read raster (GeoTIFF, TIFF, or PNG) as float tensor."""
        if fp.suffix.lower() == ".png":
            img = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
            if img is not None:
                if img.ndim == 2:
                    arr = img[np.newaxis, ...]
                elif img.ndim == 3:
                    if img.shape[2] >= 3:
                        arr = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
                    else:
                        arr = img.transpose(2, 0, 1)
                else:
                    arr = img
                return torch.from_numpy(arr.astype(np.float32)).float()
        with rasterio.open(fp) as src:
            arr = src.read()  # (C, H, W)
        return torch.from_numpy(arr).float()

    def _read_mask(self, fp: Path) -> torch.Tensor:
        """Parse RGB PNG or integer TIF mask into categorical labels {0: bg, 1: intact, 2: damaged, 99: nodata}."""
        if fp.suffix.lower() == ".png":
            img = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Failed to read mask image: {fp}")
            if img.ndim == 2:
                y = torch.from_numpy(img.astype(np.int64)).long()
            elif np.array_equal(img[..., 0], img[..., 1]) and np.array_equal(img[..., 1], img[..., 2]):
                y = torch.from_numpy(img[..., 0].astype(np.int64)).long()
            else:
                rgb = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB).astype(np.int32)
                # Standard project palette: background, intact, damaged.
                palette = np.array([[234, 207, 184], [57, 118, 175], [199, 58, 49]], dtype=np.int32)
                distances = ((rgb[..., None, :] - palette) ** 2).sum(axis=-1)
                mask = distances.argmin(axis=-1).astype(np.int64)

                # Also support common green/red damage masks and black nodata.
                r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
                mask[(g > r * 1.25) & (g > b * 1.1)] = 1
                mask[(r > g * 1.2) & (r > b * 1.2)] = 2
                mask[(r < 8) & (g < 8) & (b < 8)] = 0
                y = torch.from_numpy(mask).long()
        else:
            with rasterio.open(fp) as src:
                y = torch.from_numpy(src.read(1)).long()

        # Accept both original xBD labels and already simplified labels.
        y[(y == 3) | (y == 4)] = 2
        y[(y == 5) | (y == 6) | (y == 255)] = 99
        return y

    def _prepare_mask(self, fp: Path) -> torch.Tensor:
        """Apply the exact deterministic mask processing used during training."""
        y = self._read_mask(fp)

        # Temporarily encode nodata as class 3 because one-hot downsampling
        # cannot represent label 99 directly.
        if self.downsample_factor > 1 and y.shape[-1] > 128:
            y_for_downsampling = y.clone()
            y_for_downsampling[y_for_downsampling == 99] = 3
            y = downsample_categorical_mask(y_for_downsampling, self.downsample_factor, n_labels=4)
            y[y == 3] = 99

        if self.pixels_buffer_around_buildings > 0:
            y = apply_buffer_around_buildings(y, buffer=self.pixels_buffer_around_buildings)

        if self.task == "localization":
            y[(y > 0) & (y < 99)] = 1
        return y

    def _class_count_cache_path(self) -> Path:
        return self.data_dir / (
            f".{self.split}_mask_class_counts_v{self.CLASS_COUNT_CACHE_VERSION}_"
            f"ds{self.downsample_factor}_buffer{self.pixels_buffer_around_buildings}_{self.task}.json"
        )

    def _mask_signatures(self) -> list[dict]:
        signatures = []
        for sample in self.samples:
            path = Path(sample["mask_fp"])
            stat = path.stat()
            try:
                stored_path = str(path.relative_to(self.data_dir))
            except ValueError:
                stored_path = str(path.resolve())
            signatures.append(
                {
                    "uid": str(sample["uid"]),
                    "path": stored_path,
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
        return signatures

    def get_sample_class_counts(self) -> tuple[np.ndarray, list[str]]:
        """Load or calculate cached per-sample counts for weighted sampling."""
        class_names = ["background", "building"] if self.task == "localization" else ["background", "intact", "damaged"]
        labels = list(range(len(class_names)))
        cache_path = self._class_count_cache_path()
        signatures = self._mask_signatures()

        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if (
                    cached.get("version") == self.CLASS_COUNT_CACHE_VERSION
                    and cached.get("class_names") == class_names
                    and cached.get("mask_signatures") == signatures
                ):
                    counts = np.asarray(cached["counts"], dtype=np.int64)
                    if counts.shape == (len(self.samples), len(class_names)) and np.all(counts >= 0):
                        print(f"[SAMPLER] Loaded cached mask class counts: {cache_path}")
                        return counts, class_names
                print(f"[SAMPLER] Class-count cache is stale; recalculating: {cache_path}")
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                print(f"[SAMPLER] Could not use class-count cache ({exc}); recalculating.")

        print(f"[SAMPLER] Calculating class counts from {len(self.samples)} training masks...")
        counts = np.zeros((len(self.samples), len(class_names)), dtype=np.int64)
        for index, sample in enumerate(self.samples):
            mask = self._prepare_mask(Path(sample["mask_fp"]))
            for column, label in enumerate(labels):
                counts[index, column] = int((mask == label).sum().item())

        totals = counts.sum(axis=0)
        missing = [name for name, total in zip(class_names, totals, strict=True) if total == 0]
        if missing:
            raise ValueError(
                f"Cannot build a weighted sampler because these training-mask classes have zero pixels: {missing}. "
                "Check mask encoding before training."
            )

        payload = {
            "version": self.CLASS_COUNT_CACHE_VERSION,
            "split": self.split,
            "task": self.task,
            "downsample_factor": self.downsample_factor,
            "pixels_buffer_around_buildings": self.pixels_buffer_around_buildings,
            "class_names": class_names,
            "mask_signatures": signatures,
            "counts": counts.tolist(),
            "total_pixels_per_class": {name: int(total) for name, total in zip(class_names, totals, strict=True)},
        }
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(cache_path)
        print(f"[SAMPLER] Saved mask class counts: {cache_path}")
        return counts, class_names

    def normalize(self, x: dict) -> dict:
        """Normalize imagery per modality."""
        for k, v in x.items():
            if k.startswith("s2_tci"):
                if v.max() > 1.5:
                    x[k] = v.div(127.5).sub(1.0)
                else:
                    x[k] = v.mul(2.0).sub(1.0)
            elif k.startswith("s2"):
                min_val = self.norm_params["s2_min"]
                max_val = self.norm_params["s2_max"]
                x[k] = ((v - min_val) / (max_val - min_val + 1e-8)).clamp(0, 1)
            elif k.startswith("s1"):
                min_val = self.norm_params["s1_min"]
                max_val = self.norm_params["s1_max"]
                x[k] = ((v - min_val) / (max_val - min_val + 1e-8)).clamp(0, 1)
        return x

    def __getitem__(self, idx: int) -> dict:
        item = self.samples[idx]

        # Read images
        x = {}
        for mod in self.modalities:
            for period in self.periods:
                key = f"{mod}_{period}"
                tensor = self._read_raster(item[key])
                if mod == "s2" and len(self.s2_bands) < 12:
                    tensor = tensor[self.s2_bands_ids]
                elif mod == "s1" and len(self.s1_bands) < 2:
                    tensor = tensor[self.s1_bands_ids]
                elif mod == "s2_tci" and tensor.shape[0] > 3:
                    tensor = tensor[:3]
                if self.downsample_factor > 1 and tensor.shape[-1] > 128:
                    tensor = F.interpolate(tensor.unsqueeze(0), size=(128, 128), mode="bilinear", align_corners=False).squeeze(0)
                x[key] = tensor

        # Normalize
        x = self.normalize(x)

        # Read and process mask identically to weighted-sampler count generation.
        y = self._prepare_mask(item["mask_fp"])

        # Apply geometric augmentations
        if self.transforms is not None:
            imgs = list(x.values()) + [y]
            imgs_transformed = self.transforms(*imgs)
            x = {k: v for k, v in zip(x.keys(), imgs_transformed[:-1], strict=True)}
            y = imgs_transformed[-1]

        return {
            "images": x,
            "labels": y,
            "meta": {"xbd_uid": item["uid"], "disaster": item["disaster"]},
        }

    def get_n_imgs(self, add_predictions: bool = False) -> int:
        n_imgs = len(self.modalities) * len(self.periods) + 1
        if add_predictions:
            n_imgs += 1
        return n_imgs

    def plot(
        self,
        sample: dict,
        axs: list[mpl.axes.Axes] | None = None,
        show: bool = True,
        add_titles: bool = True,
        add_uid_as_ylabel: bool = True,
        **kwargs,
    ) -> mpl.figure.Figure:
        """Utility for visualizing a sample."""
        has_preds = "predictions" in sample
        n_imgs = self.get_n_imgs(add_predictions=has_preds)

        if axs is None:
            fig, axs = plt.subplots(1, n_imgs, figsize=(3 * n_imgs, 3))
        else:
            fig = axs[0].figure

        idx = 0
        for mod in self.modalities:
            for period in self.periods:
                ax = axs[idx]
                key = f"{mod}_{period}"
                img = sample["images"][key].cpu().numpy()
                if mod == "s2_tci":
                    img = np.clip((img + 1.0) / 2.0, 0, 1).transpose(1, 2, 0)
                    ax.imshow(img)
                elif mod == "s1":
                    ax.imshow(img[0], cmap="gray")
                else:
                    # Multi-band S2: display RGB if possible
                    if img.shape[0] >= 3:
                        rgb = img[[3, 2, 1]] if img.shape[0] >= 4 else img[:3]
                        rgb = np.clip(rgb.transpose(1, 2, 0), 0, 1)
                        ax.imshow(rgb)
                    else:
                        ax.imshow(img[0], cmap="viridis")
                if add_titles:
                    ax.set_title(key)
                ax.set_xticks([])
                ax.set_yticks([])
                idx += 1

        # Plot Ground Truth Mask
        plot_mask(sample["labels"].cpu().numpy(), ax=axs[idx], show_legend=False)
        if add_titles:
            axs[idx].set_title("Ground Truth")
        axs[idx].set_xticks([])
        axs[idx].set_yticks([])
        idx += 1

        # Plot Predictions if present
        if has_preds:
            plot_mask(sample["predictions"].cpu().numpy(), ax=axs[idx], show_legend=False)
            if add_titles:
                axs[idx].set_title("Prediction")
            axs[idx].set_xticks([])
            axs[idx].set_yticks([])

        if add_uid_as_ylabel and "meta" in sample and "xbd_uid" in sample["meta"]:
            uid = sample["meta"]["xbd_uid"]
            uid = "\n".join(uid.split("_")) + "\n"
            axs[0].set_ylabel(uid, fontsize=10, rotation=90, labelpad=10, verticalalignment="center")

        if show:
            plt.show()
        return fig