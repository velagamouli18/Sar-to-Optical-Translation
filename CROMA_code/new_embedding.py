import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from test_croma import (
    BASE_DIR,
    BATCH_SIZE,
    CROMATemporalDataset,
    DEVICE,
    IMAGE_RESOLUTION,
    NUM_WORKERS,
    SourceDataset,
    build_source_dataset,
)

from use_croma import PretrainedCROMA


# ============================================================
# CONFIGURATION
# ============================================================

REPO_DIR = Path(__file__).resolve().parent

CHECKPOINT = REPO_DIR / "CROMA_base.pt"

# NEW OUTPUT DIRECTORY.
# Existing embeddings are NOT touched.
OUTPUT_DIR = REPO_DIR / "embeddings_readme_normalized"

EXPECTED_EARTHQUAKE_TRAIN_COUNT = 135
EXPECTED_SOURCE_COUNT = 10122

EXPECTED_EMBEDDING_DIM = 1536
EXPECTED_S1_CHANNELS = 2
EXPECTED_S2_CHANNELS = 12

EPSILON = 1e-12


# ============================================================
# README NORMALIZATION
# ============================================================

def croma_readme_normalize(x):
    """
    Reproduce the CROMA README preprocessing for use_8_bit=False.

    For every channel independently:

        mean = channel mean
        std  = channel standard deviation

        min_value = mean - 2 * std
        max_value = mean + 2 * std

        img = (x - min_value) / (max_value - min_value)

        clip to [0, 1]

    Input:
        [B, C, H, W]

    Output:
        [B, C, H, W]
        float32
        range [0, 1]
    """

    x = x.float()

    normalized_channels = []

    for channel in range(x.shape[1]):

        channel_values = x[:, channel, :, :]

        min_value = (
            channel_values.mean()
            - 2 * channel_values.std()
        )

        max_value = (
            channel_values.mean()
            + 2 * channel_values.std()
        )

        denominator = max_value - min_value

        # Protect against a theoretically constant channel.
        if denominator.abs().item() < EPSILON:
            normalized = torch.zeros_like(
                channel_values
            )
        else:
            normalized = (
                channel_values - min_value
            ) / denominator

        normalized = torch.clip(
            normalized,
            0.0,
            1.0
        )

        normalized_channels.append(
            normalized.unsqueeze(dim=1)
        )

    return torch.cat(
        normalized_channels,
        dim=1
    )


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_input_tensor(
    tensor,
    modality,
    time,
    description
):

    if tensor.ndim != 4:
        raise RuntimeError(
            f"{description} {modality} {time}: "
            f"expected [B,C,H,W], got {tuple(tensor.shape)}"
        )

    expected_channels = (
        EXPECTED_S1_CHANNELS
        if modality == "S1"
        else EXPECTED_S2_CHANNELS
    )

    if tensor.shape[1] != expected_channels:
        raise RuntimeError(
            f"{description} {modality} {time}: "
            f"expected {expected_channels} channels, "
            f"got {tensor.shape[1]}"
        )

    if tensor.shape[2] != IMAGE_RESOLUTION:
        raise RuntimeError(
            f"{description} {modality} {time}: "
            f"unexpected height {tensor.shape[2]}"
        )

    if tensor.shape[3] != IMAGE_RESOLUTION:
        raise RuntimeError(
            f"{description} {modality} {time}: "
            f"unexpected width {tensor.shape[3]}"
        )

    if not torch.isfinite(tensor).all():
        raise RuntimeError(
            f"{description} {modality} {time}: "
            "NaN or Inf detected"
        )


def validate_normalized_tensor(
    tensor,
    modality,
    time,
    description
):

    minimum = tensor.min().item()
    maximum = tensor.max().item()

    if minimum < -1e-6:
        raise RuntimeError(
            f"{description} {modality} {time}: "
            f"value below 0: {minimum}"
        )

    if maximum > 1.000001:
        raise RuntimeError(
            f"{description} {modality} {time}: "
            f"value above 1: {maximum}"
        )


# ============================================================
# PRINT INPUT STATISTICS
# ============================================================

def print_input_statistics(
    tensor,
    condition,
    modality,
    time,
    description
):

    x = tensor.detach()

    print(
        f"{condition:<25} "
        f"{modality:<3} "
        f"{time:<4} "
        f"shape={tuple(x.shape)} "
        f"dtype={x.dtype} "
        f"channels={x.shape[1]} "
        f"min={x.min().item():.9f} "
        f"max={x.max().item():.9f} "
        f"mean={x.mean().item():.9f} "
        f"std={x.std(unbiased=False).item():.9f}"
    )


# ============================================================
# EMBEDDING VALIDATION
# ============================================================

def validate_joint_embedding(
    embedding,
    description
):

    if embedding.ndim != 2:
        raise RuntimeError(
            f"{description}: expected 2-D tensor, "
            f"got {tuple(embedding.shape)}"
        )

    if embedding.shape[1] != 768:
        raise RuntimeError(
            f"{description}: expected 768 dimensions, "
            f"got {embedding.shape[1]}"
        )

    if not torch.isfinite(embedding).all():
        raise RuntimeError(
            f"{description}: NaN or Inf detected"
        )


def validate_temporal_embedding(
    embedding,
    description
):

    if embedding.ndim != 2:
        raise RuntimeError(
            f"{description}: expected 2-D tensor"
        )

    if embedding.shape[1] != EXPECTED_EMBEDDING_DIM:
        raise RuntimeError(
            f"{description}: expected "
            f"{EXPECTED_EMBEDDING_DIM} dimensions, "
            f"got {embedding.shape[1]}"
        )

    if not torch.isfinite(embedding).all():
        raise RuntimeError(
            f"{description}: NaN or Inf detected"
        )

    norms = torch.linalg.norm(
        embedding,
        dim=1
    )

    if not torch.allclose(
        norms,
        torch.ones_like(norms),
        atol=1e-5
    ):
        raise RuntimeError(
            f"{description}: final embeddings "
            "are not unit-normalized"
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

    total_batches = len(dataloader)

    print()
    print("=" * 70)
    print(f"GENERATING: {description}")
    print("=" * 70)

    for batch_index, batch in enumerate(
        dataloader,
        start=1
    ):

        (
            sar_pre,
            optical_pre,
            sar_post,
            optical_post,
            uids
        ) = batch

        # ----------------------------------------------------
        # Validate tensors coming from the existing loader
        # BEFORE README normalization.
        # ----------------------------------------------------

        validate_input_tensor(
            sar_pre,
            "S1",
            "PRE",
            description
        )

        validate_input_tensor(
            optical_pre,
            "S2",
            "PRE",
            description
        )

        validate_input_tensor(
            sar_post,
            "S1",
            "POST",
            description
        )

        validate_input_tensor(
            optical_post,
            "S2",
            "POST",
            description
        )

        # ----------------------------------------------------
        # README preprocessing
        #
        # This is the ONLY input-value transformation.
        # ----------------------------------------------------

        sar_pre = croma_readme_normalize(
            sar_pre
        )

        optical_pre = croma_readme_normalize(
            optical_pre
        )

        sar_post = croma_readme_normalize(
            sar_post
        )

        optical_post = croma_readme_normalize(
            optical_post
        )

        # ----------------------------------------------------
        # Validate [0,1]
        # ----------------------------------------------------

        validate_normalized_tensor(
            sar_pre,
            "S1",
            "PRE",
            description
        )

        validate_normalized_tensor(
            optical_pre,
            "S2",
            "PRE",
            description
        )

        validate_normalized_tensor(
            sar_post,
            "S1",
            "POST",
            description
        )

        validate_normalized_tensor(
            optical_post,
            "S2",
            "POST",
            description
        )

        # ----------------------------------------------------
        # Print the actual tensors entering CROMA.
        # ----------------------------------------------------

        print_input_statistics(
            sar_pre,
            "README",
            "S1",
            "PRE",
            description
        )

        print_input_statistics(
            optical_pre,
            "README",
            "S2",
            "PRE",
            description
        )

        print_input_statistics(
            sar_post,
            "README",
            "S1",
            "POST",
            description
        )

        print_input_statistics(
            optical_post,
            "README",
            "S2",
            "POST",
            description
        )

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
        # PRE CROMA
        # ====================================================

        output_pre = model(
            SAR_images=sar_pre,
            optical_images=optical_pre
        )

        z_pre = output_pre["joint_GAP"]

        validate_joint_embedding(
            z_pre,
            f"{description} PRE joint_GAP"
        )

        # ====================================================
        # POST CROMA
        # ====================================================

        output_post = model(
            SAR_images=sar_post,
            optical_images=optical_post
        )

        z_post = output_post["joint_GAP"]

        validate_joint_embedding(
            z_post,
            f"{description} POST joint_GAP"
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

        if z_temporal.shape[1] != 1536:
            raise RuntimeError(
                f"{description}: unexpected temporal "
                f"embedding shape {tuple(z_temporal.shape)}"
            )

        # ----------------------------------------------------
        # Final L2 normalization
        # ----------------------------------------------------

        z_temporal = F.normalize(
            z_temporal,
            p=2,
            dim=1
        )

        validate_temporal_embedding(
            z_temporal,
            f"{description} temporal embedding"
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        all_embeddings.append(
            z_temporal.cpu()
        )

        all_uids.extend(
            list(uids)
        )

        if (
            batch_index == 1
            or batch_index % 10 == 0
            or batch_index == total_batches
        ):
            print(
                f"Processed batch "
                f"{batch_index}/{total_batches}"
            )

    if not all_embeddings:
        raise RuntimeError(
            f"No embeddings generated for {description}"
        )

    embeddings = torch.cat(
        all_embeddings,
        dim=0
    )

    if len(all_uids) != embeddings.shape[0]:
        raise RuntimeError(
            f"{description}: embedding/UID mismatch: "
            f"{embeddings.shape[0]} vs {len(all_uids)}"
        )

    return embeddings, all_uids


# ============================================================
# SAVE EMBEDDINGS + METADATA
# ============================================================

def save_embedding_set(
    embeddings,
    uids,
    set_name,
    expected_count
):

    print()
    print("=" * 70)
    print(f"SAVING {set_name.upper()}")
    print("=" * 70)

    if embeddings.shape != (
        expected_count,
        EXPECTED_EMBEDDING_DIM
    ):
        raise RuntimeError(
            f"{set_name}: unexpected shape "
            f"{tuple(embeddings.shape)}"
        )

    if len(uids) != expected_count:
        raise RuntimeError(
            f"{set_name}: unexpected UID count "
            f"{len(uids)}"
        )

    if len(set(uids)) != len(uids):
        raise RuntimeError(
            f"{set_name}: duplicate UIDs detected"
        )

    # --------------------------------------------------------
    # Check final norms
    # --------------------------------------------------------

    norms = torch.linalg.norm(
        embeddings,
        dim=1
    )

    if not torch.allclose(
        norms,
        torch.ones_like(norms),
        atol=1e-5
    ):
        raise RuntimeError(
            f"{set_name}: embeddings are not unit norm"
        )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    embedding_path = (
        OUTPUT_DIR
        / f"{set_name}_embeddings.npy"
    )

    metadata_path = (
        OUTPUT_DIR
        / f"{set_name}_metadata.csv"
    )

    # NEVER overwrite
    if embedding_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite:\n"
            f"{embedding_path}"
        )

    if metadata_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite:\n"
            f"{metadata_path}"
        )

    # --------------------------------------------------------
    # Save embeddings
    # --------------------------------------------------------

    array = embeddings.numpy().astype(
        np.float32
    )

    np.save(
        embedding_path,
        array
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    with metadata_path.open(
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

    # --------------------------------------------------------
    # Reload and verify saved files
    # --------------------------------------------------------

    saved = np.load(
        embedding_path
    )

    if saved.shape != array.shape:
        raise RuntimeError(
            f"{set_name}: saved embedding shape "
            f"does not match"
        )

    if saved.dtype != np.float32:
        raise RuntimeError(
            f"{set_name}: saved dtype is "
            f"{saved.dtype}, expected float32"
        )

    with metadata_path.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    if len(rows) != expected_count:
        raise RuntimeError(
            f"{set_name}: metadata has "
            f"{len(rows)} rows"
        )

    for expected_index, row in enumerate(rows):

        if int(
            row["embedding_index"]
        ) != expected_index:

            raise RuntimeError(
                f"{set_name}: metadata index "
                f"misalignment at {expected_index}"
            )

        if (
            row["xbd_uid"]
            != uids[expected_index]
        ):

            raise RuntimeError(
                f"{set_name}: UID misalignment "
                f"at {expected_index}"
            )

    print(
        f"Embedding file:\n"
        f"{embedding_path}"
    )

    print(
        f"Shape: {saved.shape}"
    )

    print(
        f"dtype: {saved.dtype}"
    )

    print(
        f"Metadata file:\n"
        f"{metadata_path}"
    )

    print(
        f"Metadata rows: {len(rows)}"
    )


# ============================================================
# OUTPUT DIRECTORY SAFETY
# ============================================================

def validate_output_directory():

    if OUTPUT_DIR.exists():

        existing_files = list(
            OUTPUT_DIR.iterdir()
        )

        if existing_files:

            raise FileExistsError(
                "Output directory already exists "
                "and is not empty:\n"
                f"{OUTPUT_DIR}\n\n"
                "Refusing to overwrite anything."
            )

    # Explicitly make sure this isn't the old
    # production embedding directory.

    old_embeddings = (
        Path(
            r"C:\Users\velaga mouli\Desktop"
            r"\Syntheticc_Project\embeddings"
        )
    ).resolve()

    if OUTPUT_DIR.resolve() == old_embeddings:

        raise RuntimeError(
            "SAFETY CHECK FAILED: "
            "output directory points to "
            "existing production embeddings."
        )


# ============================================================
# DATALOADER
# ============================================================

def make_loader(dataset):

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 78)
    print("CROMA README-NORMALIZED EMBEDDING GENERATION")
    print("=" * 78)

    print()
    print("Pipeline:")
    print()
    print("  xBD TIFF")
    print("      ↓")
    print("  existing test_croma.py loader")
    print("      ↓")
    print("  float32")
    print("      ↓")
    print("  HWC → CHW")
    print("      ↓")
    print(f"  {IMAGE_RESOLUTION} × {IMAGE_RESOLUTION} resize")
    print("      ↓")
    print("  CROMA README use_8_bit=False normalization")
    print("      ↓")
    print("  CROMA joint encoder")
    print("      ↓")
    print("  joint_GAP PRE = 768")
    print("  joint_GAP POST = 768")
    print("      ↓")
    print("  concatenate = 1536")
    print("      ↓")
    print("  final L2 normalization")
    print()
    print("No retrieval.")
    print("No S2-TCI.")
    print("No dataset modification.")
    print("Existing embeddings will NOT be overwritten.")

    # ========================================================
    # CHECK FILES
    # ========================================================

    print()
    print("=" * 78)
    print("CHECKING FILES")
    print("=" * 78)

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            f"CROMA checkpoint not found:\n"
            f"{CHECKPOINT}"
        )

    print(
        f"CROMA checkpoint: FOUND\n"
        f"{CHECKPOINT}"
    )

    validate_output_directory()

    print(
        f"\nNew output directory:\n"
        f"{OUTPUT_DIR}"
    )

    # ========================================================
    # BUILD EARTHQUAKE TRAIN
    # ========================================================

    print()
    print("=" * 78)
    print("BUILDING EARTHQUAKE TRAIN DATASET")
    print("=" * 78)

    earthquake_train_dataset = (
        CROMATemporalDataset(
            BASE_DIR / "earthquake_train_s1",
            BASE_DIR / "earthquake_train_s2"
        )
    )

    print(
        f"Earthquake TRAIN samples: "
        f"{len(earthquake_train_dataset)}"
    )

    if (
        len(earthquake_train_dataset)
        != EXPECTED_EARTHQUAKE_TRAIN_COUNT
    ):

        raise RuntimeError(
            "Unexpected earthquake TRAIN count: "
            f"{len(earthquake_train_dataset)} "
            f"(expected "
            f"{EXPECTED_EARTHQUAKE_TRAIN_COUNT})"
        )

    # ========================================================
    # BUILD SOURCE
    # ========================================================

    print()
    print("=" * 78)
    print("BUILDING SOURCE DATASET")
    print("=" * 78)

    source_entries = build_source_dataset()

    print(
        f"SOURCE samples: "
        f"{len(source_entries)}"
    )

    if (
        len(source_entries)
        != EXPECTED_SOURCE_COUNT
    ):

        raise RuntimeError(
            "Unexpected SOURCE count: "
            f"{len(source_entries)} "
            f"(expected "
            f"{EXPECTED_SOURCE_COUNT})"
        )

    source_dataset = SourceDataset(
        source_entries
    )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print()
    print("=" * 78)
    print("LOADING PRETRAINED CROMA")
    print("=" * 78)

    print(
        f"Checkpoint:\n{CHECKPOINT}"
    )

    print(
        f"Device: {DEVICE}"
    )

    model = PretrainedCROMA(
        pretrained_path=str(
            CHECKPOINT
        ),
        size="base",
        modality="both",
        image_resolution=IMAGE_RESOLUTION
    ).to(DEVICE)

    model.eval()

    print("CROMA: READY")
    print("Modality: both")
    print("Model size: base")
    print(
        f"Image resolution: "
        f"{IMAGE_RESOLUTION}"
    )

    # ========================================================
    # GENERATE EARTHQUAKE TRAIN
    # ========================================================

    earthquake_loader = make_loader(
        earthquake_train_dataset
    )

    with torch.no_grad():

        earthquake_embeddings, earthquake_uids = (
            extract_embeddings(
                earthquake_loader,
                model,
                "EARTHQUAKE TRAIN"
            )
        )

    print()
    print(
        f"Earthquake TRAIN embeddings: "
        f"{tuple(earthquake_embeddings.shape)}"
    )

    # ========================================================
    # GENERATE SOURCE
    # ========================================================

    source_loader = make_loader(
        source_dataset
    )

    with torch.no_grad():

        source_embeddings, source_uids = (
            extract_embeddings(
                source_loader,
                model,
                "SOURCE"
            )
        )

    print()
    print(
        f"SOURCE embeddings: "
        f"{tuple(source_embeddings.shape)}"
    )

    # ========================================================
    # FINAL VALIDATION BEFORE WRITING
    # ========================================================

    print()
    print("=" * 78)
    print("FINAL VALIDATION")
    print("=" * 78)

    validate_temporal_embedding(
        earthquake_embeddings,
        "Earthquake TRAIN"
    )

    validate_temporal_embedding(
        source_embeddings,
        "SOURCE"
    )

    if (
        len(earthquake_uids)
        != EXPECTED_EARTHQUAKE_TRAIN_COUNT
    ):

        raise RuntimeError(
            "Earthquake TRAIN UID count failed"
        )

    if (
        len(source_uids)
        != EXPECTED_SOURCE_COUNT
    ):

        raise RuntimeError(
            "SOURCE UID count failed"
        )

    print(
        "Earthquake TRAIN: PASSED"
    )

    print(
        "SOURCE: PASSED"
    )

    print(
        "Embedding dimension: 1536 PASSED"
    )

    print(
        "Final L2 normalization: PASSED"
    )

    print(
        "UID uniqueness: PASSED"
    )

    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=False
    )

    # ========================================================
    # SAVE
    # ========================================================

    save_embedding_set(
        earthquake_embeddings,
        earthquake_uids,
        "earthquake_train",
        EXPECTED_EARTHQUAKE_TRAIN_COUNT
    )

    save_embedding_set(
        source_embeddings,
        source_uids,
        "source",
        EXPECTED_SOURCE_COUNT
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 78)
    print("EMBEDDING GENERATION COMPLETE")
    print("=" * 78)

    print()
    print(
        "Generated new README-normalized embeddings:"
    )

    print(
        f"  Earthquake TRAIN: "
        f"{earthquake_embeddings.shape}"
    )

    print(
        f"  SOURCE: "
        f"{source_embeddings.shape}"
    )

    print()
    print(
        f"Output directory:\n"
        f"{OUTPUT_DIR}"
    )

    print()
    print("Existing production embeddings were not modified.")
    print("Dataset files were not modified.")
    print("No retrieval was performed.")
    print("No S2-TCI was used.")
    print()


if __name__ == "__main__":
    main()