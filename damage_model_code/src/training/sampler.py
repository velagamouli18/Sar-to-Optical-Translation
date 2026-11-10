"""Weighted sampler used for training to address class imbalance."""

import numpy as np
import torch
import torch.utils.data as tdata


def _sampler_from_counts(counts: np.ndarray, class_names: list[str]) -> tdata.WeightedRandomSampler:
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[0] == 0 or counts.shape[1] != len(class_names):
        raise ValueError(f"Invalid class-count matrix shape {counts.shape} for classes {class_names}")
    if not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError("Class counts must be finite and non-negative")

    totals = counts.sum(axis=0)
    missing = [name for name, total in zip(class_names, totals, strict=True) if total <= 0]
    if missing:
        raise ValueError(f"Cannot weight absent training classes: {missing}")

    total_valid = float(totals.sum())
    class_weights = total_valid / totals
    valid_per_sample = counts.sum(axis=1)
    if np.any(valid_per_sample <= 0):
        bad = np.flatnonzero(valid_per_sample <= 0).tolist()
        raise ValueError(f"Training samples contain no valid class pixels at indices: {bad[:20]}")
    sample_weights = (counts * class_weights).sum(axis=1) / valid_per_sample
    if not np.isfinite(sample_weights).all() or np.any(sample_weights <= 0):
        raise ValueError("Calculated sample weights must be finite and positive")

    totals_text = ", ".join(f"{name}={int(total)}" for name, total in zip(class_names, totals, strict=True))
    weights_text = ", ".join(f"{name}={weight:.3f}" for name, weight in zip(class_names, class_weights, strict=True))
    print(f"[SAMPLER] Training pixels: {totals_text}")
    print(f"[SAMPLER] Inverse-frequency class weights: {weights_text}")
    print(
        f"[SAMPLER] Per-sample weight range: {sample_weights.min():.4f} to {sample_weights.max():.4f}; "
        "sampling with replacement."
    )
    return tdata.WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def get_weighted_sampler(ds: tdata.Dataset, use_simplified_classes: bool = True) -> tdata.WeightedRandomSampler:
    """
    Returns a WeightedRandomSampler for the given dataset based on precomputed number of pixels per class.

    Each class is assigned a weight w_i inversely proportional to its frequency in the training set. Then each patch
    is assigned a weight based on the classes it contains.

    Args:
        ds (tdata.Dataset): The xBD12Dataset for which to compute the weights.
        use_simplified_classes (bool): Whether to use simplified damage classes (Background, Intact, Damaged) or original ones. Defaults to True.

    """

    if hasattr(ds, "get_sample_class_counts"):
        counts, class_names = ds.get_sample_class_counts()
        return _sampler_from_counts(counts, class_names)

    # Compute N_background since it's not in the geojson directly
    ds.meta["N_background"] = ds.meta["N_total"] - ds.meta[["N_intact", "N_minor", "N_major", "N_destroyed", "N_unclassified"]].sum(axis=1)

    # Get valid columns (so we exclude unclassified and no_data pixels from the weighting)
    if use_simplified_classes:
        # Merge original damage classes into a single "damaged" class
        ds.meta["N_damaged"] = ds.meta[["N_minor", "N_major", "N_destroyed"]].sum(axis=1)
        valid_class_cols = ["N_background", "N_intact", "N_damaged"]
    else:
        valid_class_cols = ["N_background", "N_intact", "N_minor", "N_major", "N_destroyed"]

    # Compute number of pixels per class for each sample
    total_px_per_class = ds.meta[valid_class_cols].sum()
    total_px = total_px_per_class.sum()  # total number of pixels per class across all samples

    # Inverse frequency weighting: we give a higher weight to classes that are less frequent in the dataset.
    weight_per_class = total_px / total_px_per_class

    # Multiply the pixel counts by the class weights to get a raw weight for each sample
    raw_row_weights = (ds.meta[valid_class_cols] * weight_per_class).sum(axis=1)

    # Number of valid pixels per sample
    valid_px_per_row = ds.meta[valid_class_cols].sum(axis=1)

    # Average weight per valid pixel for each sample
    average_weight_per_row = raw_row_weights / valid_px_per_row

    # Map weights to dataset samples
    ds.meta["uid_weights"] = average_weight_per_row  # add weights to dataset meta
    sample_weights = ds.meta["uid_weights"].to_list()

    return tdata.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)