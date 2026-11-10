import csv
import os
import numpy as np
import torch
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import tifffile as tiff
from tqdm import tqdm

from use_croma import PretrainedCROMA


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(
    os.environ.get(
        "CROMA_DATA_DIR",
        r"C:\Users\velaga mouli\Desktop\Syntheticc_Project\data\xbd_s12",
    )
)

OUTPUT_DIR = Path(
    os.environ.get(
        "CROMA_EMBEDDING_DIR",
        r"C:\Users\velaga mouli\Desktop\Syntheticc_Project\embeddings",
    )
)

CROMA_WEIGHTS = "CROMA_base.pt"

IMAGE_RESOLUTION = 120

# Start conservatively.
# On HPC we can increase this after checking VRAM.
BATCH_SIZE = 4

NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# EXPECTED DATASET COUNTS
# ============================================================

EXPECTED_COUNTS = {
    "earthquake_train": 135,
    "earthquake_val": 18,
    "earthquake_test": 40,
}


# ============================================================
# DATASET
# ============================================================

class CROMATemporalDataset(Dataset):

    def __init__(self, s1_dir, s2_dir):

        self.s1_dir = Path(s1_dir)
        self.s2_dir = Path(s2_dir)

        if not self.s1_dir.exists():
            raise FileNotFoundError(
                f"S1 directory does not exist:\n{self.s1_dir}"
            )

        if not self.s2_dir.exists():
            raise FileNotFoundError(
                f"S2 directory does not exist:\n{self.s2_dir}"
            )

        # ----------------------------------------------------
        # We use POST S1 files as the pair index.
        #
        # Every UID must then have:
        #
        # PRE  S1
        # POST S1
        # PRE  S2
        # POST S2
        # ----------------------------------------------------

        post_files = sorted(
            self.s1_dir.glob(
                "*_post_disaster_s1.tif"
            )
        )

        self.uids = []

        for path in post_files:

            uid = path.name.replace(
                "_post_disaster_s1.tif",
                ""
            )

            self.uids.append(uid)

        if len(self.uids) == 0:

            raise RuntimeError(
                f"No POST S1 files found in:\n"
                f"{self.s1_dir}"
            )

        # ----------------------------------------------------
        # Verify every UID has all four images.
        # ----------------------------------------------------

        missing = []

        for uid in self.uids:

            required_files = [

                self.s1_dir /
                f"{uid}_pre_disaster_s1.tif",

                self.s1_dir /
                f"{uid}_post_disaster_s1.tif",

                self.s2_dir /
                f"{uid}_pre_disaster_s2.tif",

                self.s2_dir /
                f"{uid}_post_disaster_s2.tif",
            ]

            for path in required_files:

                if not path.exists():

                    missing.append(str(path))

        if missing:

            message = "\n".join(missing[:20])

            raise FileNotFoundError(
                f"\nMissing required files.\n"
                f"First missing files:\n{message}"
            )

        # ----------------------------------------------------
        # Make sure UIDs are unique.
        # ----------------------------------------------------

        if len(set(self.uids)) != len(self.uids):

            raise RuntimeError(
                f"Duplicate UIDs detected in:\n"
                f"{self.s1_dir}"
            )

    def __len__(self):

        return len(self.uids)

    def _load(self, path):

        image = tiff.imread(path)

        if image.ndim != 3:

            raise ValueError(
                f"{path.name}: expected H x W x C image, "
                f"got shape {image.shape}"
            )

        # ----------------------------------------------------
        # XBD TIFF:
        #
        # H x W x C
        #
        # PyTorch:
        #
        # C x H x W
        # ----------------------------------------------------

        tensor = torch.from_numpy(
            image
        ).float()

        tensor = tensor.permute(
            2,
            0,
            1
        )

        # ----------------------------------------------------
        # Resize to CROMA input resolution.
        # ----------------------------------------------------

        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(
                IMAGE_RESOLUTION,
                IMAGE_RESOLUTION
            ),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        return tensor

    def __getitem__(self, index):

        uid = self.uids[index]

        # ----------------------------------------------------
        # PRE
        # ----------------------------------------------------

        sar_pre = self._load(
            self.s1_dir /
            f"{uid}_pre_disaster_s1.tif"
        )

        optical_pre = self._load(
            self.s2_dir /
            f"{uid}_pre_disaster_s2.tif"
        )

        # ----------------------------------------------------
        # POST
        # ----------------------------------------------------

        sar_post = self._load(
            self.s1_dir /
            f"{uid}_post_disaster_s1.tif"
        )

        optical_post = self._load(
            self.s2_dir /
            f"{uid}_post_disaster_s2.tif"
        )

        return (
            sar_pre,
            optical_pre,
            sar_post,
            optical_post,
            uid
        )


# ============================================================
# EMBEDDING EXTRACTION
# ============================================================

@torch.no_grad()
def extract_embeddings(
    dataloader,
    model,
    description
):

    all_embeddings = []
    all_uids = []

    for (
        sar_pre,
        optical_pre,
        sar_post,
        optical_post,
        uids
    ) in tqdm(
        dataloader,
        desc=description
    ):

        # ----------------------------------------------------
        # Move to device
        # ----------------------------------------------------

        sar_pre = sar_pre.to(
            DEVICE,
            non_blocking=True
        )

        optical_pre = optical_pre.to(
            DEVICE,
            non_blocking=True
        )

        sar_post = sar_post.to(
            DEVICE,
            non_blocking=True
        )

        optical_post = optical_post.to(
            DEVICE,
            non_blocking=True
        )

        # ====================================================
        # PRE EMBEDDING
        # ====================================================

        output_pre = model(
            SAR_images=sar_pre,
            optical_images=optical_pre
        )

        z_pre = output_pre["joint_GAP"]

        if z_pre.ndim != 2 or z_pre.shape[1] != 768:

            raise RuntimeError(
                f"Unexpected PRE embedding shape: "
                f"{tuple(z_pre.shape)}"
            )

        # ====================================================
        # POST EMBEDDING
        # ====================================================

        output_post = model(
            SAR_images=sar_post,
            optical_images=optical_post
        )

        z_post = output_post["joint_GAP"]

        if z_post.ndim != 2 or z_post.shape[1] != 768:

            raise RuntimeError(
                f"Unexpected POST embedding shape: "
                f"{tuple(z_post.shape)}"
            )

        # ====================================================
        # BI-TEMPORAL EMBEDDING
        # ====================================================

        z_temporal = torch.cat(
            [
                z_pre,
                z_post
            ],
            dim=1
        )

        # Expected:
        #
        # [B, 1536]
        #

        if z_temporal.shape[1] != 1536:

            raise RuntimeError(
                f"Unexpected temporal embedding shape: "
                f"{tuple(z_temporal.shape)}"
            )

        # ----------------------------------------------------
        # Normalize ONLY the final temporal embedding.
        #
        # IMPORTANT:
        #
        # No mean centering.
        # No similarity.
        # No retrieval.
        # ----------------------------------------------------

        z_temporal = F.normalize(
            z_temporal,
            p=2,
            dim=1
        )

        all_embeddings.append(
            z_temporal.cpu()
        )

        all_uids.extend(
            list(uids)
        )

    if not all_embeddings:

        raise RuntimeError(
            f"No embeddings were generated for {description}"
        )

    embeddings = torch.cat(
        all_embeddings,
        dim=0
    )

    return embeddings, all_uids


# ============================================================
# SAVE EMBEDDINGS
# ============================================================

def save_embeddings(
    embeddings,
    uids,
    output_dir,
    set_name
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # NPY
    # --------------------------------------------------------

    embedding_path = (
        output_dir /
        f"{set_name}_embeddings.npy"
    )

    np.save(
        embedding_path,
        embeddings.numpy().astype(
            np.float32
        )
    )

    # --------------------------------------------------------
    # UID metadata
    # --------------------------------------------------------

    metadata_path = (
        output_dir /
        f"{set_name}_metadata.csv"
    )

    with open(
        metadata_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "embedding_index",
                "xbd_uid"
            ]
        )

        for index, uid in enumerate(uids):

            writer.writerow(
                [
                    index,
                    uid
                ]
            )

    print(
        f"\nSaved {set_name}:"
    )

    print(
        f"  Embeddings: {embedding_path}"
    )

    print(
        f"  Metadata:   {metadata_path}"
    )

    print(
        f"  Shape:      {tuple(embeddings.shape)}"
    )


# ============================================================
# VERIFY EMBEDDINGS
# ============================================================

def verify_embeddings(
    embeddings,
    uids,
    expected_count,
    name
):

    print(
        f"\nVerifying {name}..."
    )

    # Count

    assert len(uids) == expected_count, (
        f"{name}: expected {expected_count} UIDs, "
        f"got {len(uids)}"
    )

    # Shape

    assert embeddings.shape == (
        expected_count,
        1536
    ), (
        f"{name}: expected shape "
        f"({expected_count}, 1536), "
        f"got {tuple(embeddings.shape)}"
    )

    # Dtype

    assert embeddings.dtype == torch.float32

    # NaN

    assert not torch.isnan(
        embeddings
    ).any()

    # Inf

    assert not torch.isinf(
        embeddings
    ).any()

    # Norm

    norms = torch.linalg.norm(
        embeddings,
        dim=1
    )

    assert torch.allclose(
        norms,
        torch.ones_like(norms),
        atol=1e-5
    )

    # Unique UIDs

    assert len(set(uids)) == len(uids), (
        f"{name}: duplicate UIDs detected"
    )

    print(
        f"  Count:       {len(uids)}"
    )

    print(
        f"  Dimension:   {embeddings.shape[1]}"
    )

    print(
        f"  dtype:       {embeddings.dtype}"
    )

    print(
        f"  NaN:         "
        f"{torch.isnan(embeddings).any().item()}"
    )

    print(
        f"  Inf:         "
        f"{torch.isinf(embeddings).any().item()}"
    )

    print(
        f"  Norm mean:   "
        f"{norms.mean().item():.8f}"
    )

    print(
        f"  Norm min:    "
        f"{norms.min().item():.8f}"
    )

    print(
        f"  Norm max:    "
        f"{norms.max().item():.8f}"
    )

    print(
        "  PASSED"
    )


# ============================================================
# PROCESS NORMAL SPLIT
# ============================================================

def process_split(
    name,
    s1_folder,
    s2_folder,
    expected_count
):

    print(
        "\n" + "=" * 70
    )

    print(
        f"PROCESSING {name}"
    )

    print(
        "=" * 70
    )

    s1_dir = BASE_DIR / s1_folder
    s2_dir = BASE_DIR / s2_folder

    print(
        f"S1: {s1_dir}"
    )

    print(
        f"S2: {s2_dir}"
    )

    dataset = CROMATemporalDataset(
        s1_dir,
        s2_dir
    )

    print(
        f"Pairs found: {len(dataset)}"
    )

    assert len(dataset) == expected_count, (
        f"{name}: expected {expected_count}, "
        f"found {len(dataset)}"
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    embeddings, uids = extract_embeddings(
        loader,
        model,
        name
    )

    verify_embeddings(
        embeddings,
        uids,
        expected_count,
        name
    )

    save_embeddings(
        embeddings,
        uids,
        OUTPUT_DIR,
        name
    )

    return embeddings, uids


# ============================================================
# BUILD SOURCE DATASET
# ============================================================

class SourceDataset(Dataset):

    def __init__(self, entries):

        self.entries = entries

        if len(self.entries) == 0:

            raise RuntimeError(
                "No source pairs were found."
            )

        # ----------------------------------------------------
        # Verify no duplicate UID.
        # ----------------------------------------------------

        uids = [
            entry[0]
            for entry in self.entries
        ]

        if len(set(uids)) != len(uids):

            raise RuntimeError(
                "Duplicate source UIDs detected."
            )

        # ----------------------------------------------------
        # Verify all four images.
        # ----------------------------------------------------

        missing = []

        for uid, s1_dir, s2_dir in self.entries:

            required = [

                s1_dir /
                f"{uid}_pre_disaster_s1.tif",

                s1_dir /
                f"{uid}_post_disaster_s1.tif",

                s2_dir /
                f"{uid}_pre_disaster_s2.tif",

                s2_dir /
                f"{uid}_post_disaster_s2.tif",
            ]

            for path in required:

                if not path.exists():

                    missing.append(str(path))

        if missing:

            raise FileNotFoundError(
                "Missing source images:\n"
                + "\n".join(missing[:20])
            )

    def __len__(self):

        return len(self.entries)

    def _load(self, path):

        image = tiff.imread(path)

        if image.ndim != 3:

            raise ValueError(
                f"{path.name}: expected H x W x C, "
                f"got {image.shape}"
            )

        tensor = torch.from_numpy(
            image
        ).float().permute(
            2,
            0,
            1
        )

        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(
                IMAGE_RESOLUTION,
                IMAGE_RESOLUTION
            ),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        return tensor

    def __getitem__(self, index):

        uid, s1_dir, s2_dir = (
            self.entries[index]
        )

        sar_pre = self._load(
            s1_dir /
            f"{uid}_pre_disaster_s1.tif"
        )

        optical_pre = self._load(
            s2_dir /
            f"{uid}_pre_disaster_s2.tif"
        )

        sar_post = self._load(
            s1_dir /
            f"{uid}_post_disaster_s1.tif"
        )

        optical_post = self._load(
            s2_dir /
            f"{uid}_post_disaster_s2.tif"
        )

        return (
            sar_pre,
            optical_pre,
            sar_post,
            optical_post,
            uid
        )


# ============================================================
# BUILD SOURCE UID LIST
# ============================================================

def build_source_dataset():

    print(
        "\n" + "=" * 70
    )

    print(
        "BUILDING SOURCE DATASET"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Earthquake UIDs are identified from filenames.
    #
    # We include train + val + test because ALL earthquake
    # samples must be excluded from the source pool.
    # --------------------------------------------------------

    earthquake_uids = set()

    for folder in [
        "train_s1",
        "val_s1",
        "test_s1"
    ]:

        folder_path = BASE_DIR / folder

        for path in folder_path.glob(
            "*_post_disaster_s1.tif"
        ):

            uid = path.name.replace(
                "_post_disaster_s1.tif",
                ""
            )

            if "earthquake" in uid.lower():

                earthquake_uids.add(uid)

    print(
        f"Earthquake UIDs excluded: "
        f"{len(earthquake_uids)}"
    )

    assert len(earthquake_uids) == 193, (
        f"Expected 193 earthquake UIDs, "
        f"found {len(earthquake_uids)}"
    )

    # --------------------------------------------------------
    # Source = every non-earthquake pair from train + val.
    #
    # Test is NEVER used as source.
    # --------------------------------------------------------

    entries = []

    for folder in [
        "train_s1",
        "val_s1"
    ]:

        s1_dir = BASE_DIR / folder

        s2_dir = BASE_DIR / folder.replace(
            "_s1",
            "_s2"
        )

        for path in sorted(
            s1_dir.glob(
                "*_post_disaster_s1.tif"
            )
        ):

            uid = path.name.replace(
                "_post_disaster_s1.tif",
                ""
            )

            # Exclude all earthquake samples.

            if uid in earthquake_uids:

                continue

            entries.append(
                (
                    uid,
                    s1_dir,
                    s2_dir
                )
            )

    # --------------------------------------------------------
    # Remove duplicates.
    # --------------------------------------------------------

    unique_entries = {}

    for uid, s1_dir, s2_dir in entries:

        if uid in unique_entries:

            raise RuntimeError(
                f"Duplicate source UID found: {uid}"
            )

        unique_entries[uid] = (
            uid,
            s1_dir,
            s2_dir
        )

    entries = [
        unique_entries[uid]
        for uid in sorted(unique_entries)
    ]

    print(
        f"Source pairs: {len(entries)}"
    )

    # --------------------------------------------------------
    # Expected:
    #
    # Total remaining after frozen earthquake test:
    # 10275
    #
    # Earthquake remaining:
    # 153
    #
    # Therefore non-earthquake source:
    #
    # 10275 - 153 = 10122
    #
    # --------------------------------------------------------

    assert len(entries) == 10122, (
        f"Expected 10122 non-earthquake source pairs, "
        f"found {len(entries)}"
    )

    print(
        "Source dataset count: PASSED"
    )

    return entries


# ============================================================
# MAIN
# ============================================================

def main():

    global model

    print(
        "=" * 70
    )

    print(
        "CROMA BI-TEMPORAL EMBEDDING EXTRACTION"
    )

    print(
        "=" * 70
    )

    print(
        f"\nDevice: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "CUDA:",
            torch.version.cuda
        )

    print(
        f"\nBase directory:\n{BASE_DIR}"
    )

    print(
        f"\nOutput directory:\n{OUTPUT_DIR}"
    )

    # ========================================================
    # CHECK DATASET STRUCTURE
    # ========================================================

    required_dirs = [

        # Original diffusion dataset

        "train_s1",
        "train_s2",

        "val_s1",
        "val_s2",

        "test_s1",
        "test_s2",

        # Dedicated earthquake embedding datasets

        "earthquake_train_s1",
        "earthquake_train_s2",

        "earthquake_val_s1",
        "earthquake_val_s2",
    ]

    print(
        "\nChecking required directories..."
    )

    for folder in required_dirs:

        path = BASE_DIR / folder

        if not path.exists():

            raise FileNotFoundError(
                f"Required directory missing:\n{path}"
            )

    print(
        "Directory structure: PASSED"
    )

    # ========================================================
    # LOAD CROMA
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "LOADING CROMA"
    )

    print(
        "=" * 70
    )

    model = PretrainedCROMA(
        pretrained_path=CROMA_WEIGHTS,
        size="base",
        modality="both",
        image_resolution=IMAGE_RESOLUTION
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        "\nCROMA loaded."
    )

    print(
        "Pre embedding:  768-D"
    )

    print(
        "Post embedding: 768-D"
    )

    print(
        "Temporal:       1536-D"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # EARTHQUAKE TRAIN
    # ========================================================

    process_split(
        "earthquake_train",
        "earthquake_train_s1",
        "earthquake_train_s2",
        135
    )

    # ========================================================
    # EARTHQUAKE VALIDATION
    # ========================================================

    process_split(
        "earthquake_val",
        "earthquake_val_s1",
        "earthquake_val_s2",
        18
    )

    # ========================================================
    # EARTHQUAKE TEST
    # ========================================================

    process_split(
        "earthquake_test",
        "test_s1",
        "test_s2",
        40
    )

    # ========================================================
    # SOURCE DATASET
    # ========================================================

    source_entries = build_source_dataset()

    source_dataset = SourceDataset(
        source_entries
    )

    print(
        f"Source dataset length: "
        f"{len(source_dataset)}"
    )

    source_loader = DataLoader(
        source_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    # ========================================================
    # SOURCE EMBEDDINGS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "EXTRACTING SOURCE EMBEDDINGS"
    )

    print(
        "=" * 70
    )

    source_embeddings, source_uids = (
        extract_embeddings(
            source_loader,
            model,
            "source"
        )
    )

    # ========================================================
    # VERIFY SOURCE
    # ========================================================

    verify_embeddings(
        source_embeddings,
        source_uids,
        len(source_entries),
        "source"
    )

    # ========================================================
    # SAVE SOURCE
    # ========================================================

    save_embeddings(
        source_embeddings,
        source_uids,
        OUTPUT_DIR,
        "source"
    )

    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "EMBEDDING EXTRACTION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nCreated:"
    )

    print(
        "  earthquake_train_embeddings.npy"
    )

    print(
        "  earthquake_train_metadata.csv"
    )

    print(
        "  earthquake_val_embeddings.npy"
    )

    print(
        "  earthquake_val_metadata.csv"
    )

    print(
        "  earthquake_test_embeddings.npy"
    )

    print(
        "  earthquake_test_metadata.csv"
    )

    print(
        "  source_embeddings.npy"
    )

    print(
        "  source_metadata.csv"
    )

    print(
        "\nEmbedding dimensions:"
    )

    print(
        "  PRE  = 768-D"
    )

    print(
        "  POST = 768-D"
    )

    print(
        "  TEMPORAL = 1536-D"
    )

    print(
        "\nNo mean-centering."
    )

    print(
        "No similarity calculation."
    )

    print(
        "No retrieval."
    )

    print(
        "No top-K."
    )

    print(
        "No threshold."
    )

    print(
        "\nThe embedding bank is ready."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
