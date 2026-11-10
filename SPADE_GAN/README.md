# Run the 12-band SPADE GAN training

## 1. Install dependencies

Use Python 3.10 or newer. Install PyTorch and torchvision builds appropriate for your CPU or CUDA setup, then install the training packages:

```powershell
python -m pip install torch torchvision
python -m pip install numpy pandas matplotlib pillow tifffile tqdm scikit-image
```

The first run may download pretrained VGG-19 weights from torchvision.

## 2. Prepare the dataset

Set `--dataset_path` to a folder containing these split folders:

```text
dataset/
├── train_s1/
├── train_s2/
├── val_s1/
└── val_s2/
```

Each split needs matching pre- and post-disaster files for every sample UID. Use names such as `<uid>_pre_disaster_s1.tif`, `<uid>_post_disaster_s1.tif`, `<uid>_pre_disaster_s2.tif`, and `<uid>_post_disaster_s2.tif`. S1 inputs must contain two channels; S2 inputs must contain all 12 bands.

## 3. Run training

From the `SPADE_GAN` folder, run this in PowerShell. Replace the dataset path with your actual dataset location:

```powershell
python train_bitemporal_spade.py `
    --dataset_path "C:\path\to\xbd_s12_experiment_prepost" `
    --optical-mode s2 `
    --batch_size 16 `
    --epochs 100 `
    --decay_epoch 50 `
    --lr_g 1e-4 `
    --lr_d 4e-4 `
    --lambda_l1 10.0 `
    --lambda_vgg 10.0 `
    --lambda_change 5.0 `
    --lambda_color 1.0 `
    --lambda_tv 0.1 `
    --lambda_sam 1.0 `
    --checkpoint-interval 5 `
    --output_dir "checkpoints/bitemporal_spade"
```

Checkpoints and training outputs are saved under `checkpoints/bitemporal_spade`. 
