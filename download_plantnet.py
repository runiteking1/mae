"""
Usage:
    module load python
    export HTTPS_PROXY=http://proxy.ca.sandia.gov:80
    export HTTP_PROXY=http://proxy.ca.sandia.gov:80
    uv run python download_plantnet.py
"""
from datasets import load_dataset

print('Running')
for split in ("train", "validation", "test"):
    print(f"Downloading split: {split} ...")
    ds = load_dataset("mikehemberger/plantnet300K", split=split)
    print(f"  {split}: {len(ds):,} examples")

print("Done. Dataset cached; future runs can use HF_DATASETS_OFFLINE=1.")

