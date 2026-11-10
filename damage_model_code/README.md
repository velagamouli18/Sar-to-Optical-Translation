# Siamese U-Net damage model

This folder contains the Siamese U-Net training pipeline configured for the full 12-band Sentinel-2 imagery. The Siamese branches receive the pre-event and post-event images, each with 12 S2 bands. The source repository and dataset files are not included.

## Install

Use Python 3.10 or newer. Install a PyTorch and torchvision build that matches your CPU or CUDA setup, then install the remaining requirements:

```bash
python -m pip install -r requirements.txt
```

## Dataset layout

Pass the root folder containing the split folders as `data.data_dir`. The default configuration uses Sentinel-2 only and reads all 12 bands:

```text
dataset/
├── train_s2/           # pre and post 12-band images
├── train_masks/
├── val_s2/
├── val_masks/
├── test_s2/             # optional
└── test_masks/          # optional
```

The loader pairs files by sample UID. Image names should identify pre- and post-event images, for example `<uid>_pre_disaster_s2.tif` and `<uid>_post_disaster_s2.tif`. TIFF inputs are recommended for 12-band data. Masks should use `<uid>_mask.tif` or `.png` and encode background, intact, and damaged classes; unclassified/no-data pixels are ignored by the loss.

## Train

From this folder, run:

```bash
python -m src.training.main run_name=siamese_s2_12band data.data_dir="/path/to/dataset" logger=local
```

The trainer infers the input channel count from the dataset, so the model receives 12 channels per Siamese branch. To train with both Sentinel-1 and Sentinel-2 instead, override `data.modalities=[s1,s2]`.

Checkpoints and run outputs are written under `logs/<run_name>` unless `output_dir` is set. Set `training.sampler_train=null` to disable weighted sampling.
