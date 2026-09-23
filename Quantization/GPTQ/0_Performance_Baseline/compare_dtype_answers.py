import torch
import glob
import os
import pandas as pd

OUT_DIR = "./output"


def latest_ckpt_per_dtype(dataset_name):
    # Use only the single .pt file with the most recent mtime per dtype (grouped by the stored dtype field instead of the filename - safer)
    files = glob.glob(os.path.join(OUT_DIR, f"{dataset_name}_*.pt"))
    latest = {}
    for f in files:
        if f.endswith("_all_done.pt"):
            continue
        ck = torch.load(f, map_location="cpu")
        dtype = ck.get("dtype")
        if dtype is None:
            continue
        mtime = os.path.getmtime(f)
        if dtype not in latest or mtime > latest[dtype][0]:
            latest[dtype] = (mtime, f, ck)
    return {dtype: ck for dtype, (mtime, f, ck) in latest.items()}


def build_and_save(cells_by_dtype, out_name, dataset_name):
    lengths = {dtype: len(cells) for dtype, cells in cells_by_dtype.items()}
    min_len = min(lengths.values())
    if len(set(lengths.values())) > 1:
        print(f"[warning] {dataset_name}: sample counts differ by dtype {lengths} - truncating to the shortest length ({min_len})")
    df = pd.DataFrame({dtype: cells[:min_len] for dtype, cells in cells_by_dtype.items()})
    df.index.name = "input_idx"
    out_path = os.path.join(OUT_DIR, out_name)
    df.to_csv(out_path)
    print(f"Saved {out_path} ({df.shape[0]} rows x {df.shape[1]} dtypes: {list(cells_by_dtype.keys())})")


# === FineWeb: cell = NLL value ===
fineweb_ckpts = latest_ckpt_per_dtype("fineweb")
if fineweb_ckpts:
    build_and_save(
        {dtype: ck["fineweb_NLLs"] for dtype, ck in fineweb_ckpts.items()},
        "fineweb_NLL_by_dtype.csv",
        "fineweb",
    )
else:
    print("could not find fineweb .pt file")

# === HellaSwag: cell = predicted answer (index with the smallest NLL among the 4 candidates) ===
hellaswag_ckpts = latest_ckpt_per_dtype("hellaswag")
if hellaswag_ckpts:
    pred_by_dtype = {}
    for dtype, ck in hellaswag_ckpts.items():
        candidate_nlls = ck["candidate_NLLs"]  # length-4 list per sample
        pred_by_dtype[dtype] = [int(torch.tensor(sample_nlls).argmin()) for sample_nlls in candidate_nlls]
    build_and_save(pred_by_dtype, "hellaswag_predicted_label_by_dtype.csv", "hellaswag")
else:
    print("could not find hellaswag .pt file")
