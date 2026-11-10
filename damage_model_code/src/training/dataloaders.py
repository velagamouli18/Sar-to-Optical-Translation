"""Contains the logic to create the dataloaders."""

import torch.utils.data as tdata

from src.training.augmentation import RandomFlipRotate90
from src.training.sampler import get_weighted_sampler


def modalities_from_input_pair(input_pair: str | None, modalities: str | list | None = None) -> list[str]:
    """Resolve a convenient CLI input-pair name into dataset modalities."""
    if input_pair is None:
        resolved = ["s1", "s2"] if modalities is None else modalities
        return [resolved] if isinstance(resolved, str) else list(resolved)

    normalized = input_pair.lower().replace("+", "_").replace(",", "_").replace("-", "_")
    choices = {
        "s1_s2": ["s1", "s2"],
        "s1_s2_tci": ["s1", "s2_tci"],
    }
    if normalized not in choices:
        raise ValueError(f"Unknown input_pair '{input_pair}'. Choose 's1_s2' or 's1_s2_tci'.")
    return choices[normalized]


def get_dataloaders(
    modalities: str | list | None = None,
    task: str = "multiclass",
    which_split: str = "event",
    batch_size: int = 8,
    num_workers: int = 8,
    fraction_valid: float = 0.15,
    sampler_train: str | None = "weighted",
    use_transforms: bool = True,
    pixels_buffer_around_buildings: int = 3,
    **kwargs,
) -> dict[str, tdata.DataLoader]:
    """
    Create dataloaders for training, validation and testing.

    See `xBDS12Dataset` for more details on the dataset and the available options.

    Args:
        modalities (str | list): The modalities to use. Defaults to ["s1", "s2"].
        task (str): The task to perform, either localization or multiclass. Defaults to "multiclass".
        which_split (str): The data split to use, either event, xview2 or full. Defaults to "event".
        batch_size (int): The batch size. Defaults to 8.
        num_workers (int): The number of workers for data loading. Defaults to 8.
        fraction_valid (float): The fraction of data to use for validation. Must be between 0 and 1. Defaults to 0.15.
        sampler_train (str | None): The sampler to use for training. Can be either 'weighted' or None. Defaults to weighted.
        use_transforms (bool): Whether to use data augmentation transforms. Defaults to True.
        pixels_buffer_around_buildings (int): The number of pixels to buffer around buildings for training augmentation. Defaults to 3.
        **kwargs: Additional arguments to pass to the dataset.

    Returns:
        dict[str, tdata.DataLoader]: A dictionary containing the dataloaders.
    """

    print(f"Creating all dataloaders for the {which_split} split...")

    assert 0 <= fraction_valid < 1, "fraction_valid must be in [0, 1)"
    input_pair = kwargs.pop("input_pair", None)
    modalities = modalities_from_input_pair(input_pair, modalities)

    print("Creating dataloaders...")
    print(f"{which_split=}")
    print(f"{modalities=}")
    print(f"{task=}")

    # For efficient data loading
    dataloaders_kwargs = {
        "batch_size": batch_size,
        "pin_memory": False,
        "num_workers": num_workers,
        "persistent_workers": False,
        "prefetch_factor": 2 if num_workers > 0 else None,
    }

    # Transforms
    if use_transforms:
        transforms = RandomFlipRotate90()
    else:
        transforms = None

    data_dir = kwargs.pop("data_dir", None)
    if data_dir is not None:
        from src.training.folder_dataset import FolderXBDS12Dataset

        print(f"Loading dataset from custom folder: {data_dir}")
        ds_train = FolderXBDS12Dataset(
            data_dir=data_dir,
            split="train",
            modalities=modalities,
            transforms=transforms,
            task=task,
            pixels_buffer_around_buildings=pixels_buffer_around_buildings,
            **kwargs,
        )

        if sampler_train == "weighted":
            print("Using weighted sampling for training")
            sampler = get_weighted_sampler(ds_train)
            train_loader = tdata.DataLoader(ds_train, sampler=sampler, drop_last=True, **dataloaders_kwargs)
        else:
            if sampler_train is not None:
                raise ValueError("sampler_train must be either 'weighted' or None")
            train_loader = tdata.DataLoader(ds_train, shuffle=True, drop_last=True, **dataloaders_kwargs)

        ds_valid = FolderXBDS12Dataset(
            data_dir=data_dir,
            split="val",
            modalities=modalities,
            transforms=None,
            task=task,
            pixels_buffer_around_buildings=0,
            **kwargs,
        )
        valid_loader = tdata.DataLoader(ds_valid, shuffle=False, **dataloaders_kwargs)

        try:
            ds_test = FolderXBDS12Dataset(
                data_dir=data_dir,
                split="test",
                modalities=modalities,
                transforms=None,
                task=task,
                pixels_buffer_around_buildings=0,
                **kwargs,
            )
            test_loader = tdata.DataLoader(ds_test, shuffle=False, **dataloaders_kwargs) if len(ds_test) > 0 else None
        except Exception as e:
            print(f"Warning: Could not create test dataloader: {e}")
            test_loader = None

        return {"train": train_loader, "valid": valid_loader, "test": test_loader}

    # Check if specific disasters are requested
    disaster_train = kwargs.pop("disaster_train", None)
    disaster_test = kwargs.pop("disaster_test", None)

    # Create datasets and dataloaders
    from src.training.dataset import xBDS12Dataset

    # Training dataloader (with augmentation, buffer around buildings, and weighted sampling)
    ds_train = xBDS12Dataset(
        split="train",
        which_split=which_split,
        modalities=modalities,
        disasters=disaster_train,
        transforms=transforms,
        task=task,
        fraction_valid=fraction_valid,
        pixels_buffer_around_buildings=pixels_buffer_around_buildings,  # buffer for training
        **kwargs,
    )
    if sampler_train == "weighted":
        print("Using weighted sampling for training")
        sampler_train = get_weighted_sampler(ds_train)
        train_loader = tdata.DataLoader(
            ds_train,
            sampler=sampler_train,
            drop_last=True,
            **dataloaders_kwargs,
        )
    else:
        assert sampler_train is None, "sampler_train must be either 'weighted' or None"
        train_loader = tdata.DataLoader(ds_train, shuffle=True, drop_last=True, **dataloaders_kwargs)

    # Validation dataloader
    if fraction_valid > 0:
        ds_valid = xBDS12Dataset(
            split="valid",
            which_split=which_split,
            modalities=modalities,
            disasters=disaster_train,
            transforms=None,  # no augmentation for validation
            task=task,
            fraction_valid=fraction_valid,
            pixels_buffer_around_buildings=0,  # no buffer for validation
            **kwargs,
        )
        valid_loader = tdata.DataLoader(ds_valid, shuffle=False, **dataloaders_kwargs)
    else:
        valid_loader = None

    # Test dataloader
    if which_split != "full" or disaster_test is not None:
        ds_test = xBDS12Dataset(
            split="test",
            which_split=which_split,
            modalities=modalities,
            disasters=disaster_test,
            transforms=None,  # no augmentation for testing
            task=task,
            pixels_buffer_around_buildings=0,  # no buffer for testing
            **kwargs,
        )
        test_loader = tdata.DataLoader(ds_test, shuffle=False, **dataloaders_kwargs)
    else:
        test_loader = None

    return {"train": train_loader, "valid": valid_loader, "test": test_loader}


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from src.training.utils import unbind_samples

    # Example usage
    batch_size = 4
    dataloaders = get_dataloaders(
        modalities=["s1", "s2_tci", "xbd"],
        task="multiclass",
        which_split="event",
        batch_size=batch_size,
        num_workers=4,
        fraction_valid=0.1,
        sampler_train="weighted",
        use_transforms=True,
        pixels_buffer_around_buildings=5,
    )
    dl = dataloaders["train"]

    for _batch in dl:
        break

    samples = unbind_samples(_batch)
    n_imgs = dl.dataset.get_n_imgs()
    fig, axs = plt.subplots(batch_size, n_imgs, figsize=(3 * n_imgs, 12))
    for i, sample in enumerate(samples):
        dl.dataset.plot(sample, axs=axs[i])
    plt.tight_layout()
    plt.show()