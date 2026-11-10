# CROMA embedding and retrieval code

This folder contains the CROMA embedding-generation pipeline, its model/data-loader dependencies, and a cosine-similarity retriever. It excludes the pretrained model weights, source imagery, generated embedding arrays, and analysis outputs.

## Contents

- `new_embedding.py`: creates normalized 1,536-dimensional bi-temporal embeddings and aligned UID CSV files.
- `test_croma.py`: dataset definitions and source-pool construction imported by the generator.
- `use_croma.py`: pretrained CROMA model implementation.
- `retrieve_embeddings.py`: retrieves top-k nearest embeddings using cosine similarity.

## Setup

Install a PyTorch build appropriate for your machine, then install the other dependencies:

```powershell
python -m pip install -r requirements.txt
```

Put the pretrained `CROMA_base.pt` checkpoint in this folder. The generator reads it from beside `new_embedding.py`.

Set the dataset and output locations before generating embeddings. The dataset must follow the folder layout expected by `test_croma.py`: `earthquake_train_s1`, `earthquake_train_s2`, and the standard `train_s1`, `train_s2`, `val_s1`, `val_s2`, `test_s1`, and `test_s2` folders. Filenames are paired by UID and must contain the expected pre/post-disaster Sentinel-1 and Sentinel-2 TIFF files.

```powershell
$env:CROMA_DATA_DIR = "C:\path\to\xbd_s12"
$env:CROMA_EMBEDDING_DIR = "C:\path\to\embedding_output"
python new_embedding.py
```


## Retrieve similar embeddings

After generation, query earthquake-train embeddings against the source pool:

```powershell
python retrieve_embeddings.py `
  --query "embeddings_readme_normalized\earthquake_train_embeddings.npy" `
  --query-metadata "embeddings_readme_normalized\earthquake_train_metadata.csv" `
  --database "embeddings_readme_normalized\source_embeddings.npy" `
  --database-metadata "embeddings_readme_normalized\source_metadata.csv" `
  --top-k 5 `
  --output "retrieval_results.csv"
```

The result CSV contains one row per query match, ordered by descending cosine similarity. Embeddings are normalized by the retriever before scoring, so the search also works if arrays are not already unit-normalized.
