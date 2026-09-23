# Quantization

Investigates how much post-training quantization degrades nanoGPT's GPT-2 — by
first establishing how far dtype precision alone (fp64 → fp32 → fp16/bf16 → int8,
a plain `.to(dtype=...)` cast, not a real quantization scheme yet) can be pushed
before HellaSwag accuracy/NLL and FineWeb-val perplexity/NLL visibly drop, then
applying actual quantization from whatever point that degradation starts.

This is part of the broader [AI efficiency study](../../../README.md) in this repository
and is written up in more detail in the accompanying blog post.

**Status: work in progress.** The dtype-precision sweep and the two evaluation
datasets below are in place and being run; real quantization (int8/int4 via a
scale/zero-point scheme or a library like bitsandbytes, not a naive dtype cast)
hasn't been applied yet, and results/conclusions are still being collected.

## What was changed vs. original nanoGPT

`model.py` is [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)
`model.py`, with `CausalSelfAttention` hardcoded to `self.flash = False` — always
using the manual/"math" attention implementation instead of
`scaled_dot_product_attention`, so the same kernel path is used regardless of
dtype and results stay comparable across the sweep (same approach as
[`Memory measurement/`](../../../Memory%20measurement/README.md)).

dtype is applied directly to the model weights via `model.to(device, dtype=...)`,
not `torch.autocast` — so parameters and activations run end-to-end in that dtype,
rather than PyTorch keeping fp32 master weights under the hood.

`hellaswag_evaluation.py`'s scoring logic (load candidates → compute per-candidate
NLL → pick the argmin) is adapted from the dtype-sweep scripts in
[`../Kernel investigation/Evaluation/`](../../../Kernel%20investigation/README.md),
restructured to run one dtype per pass instead of sweeping dtype/batch/padding
for kernel-selection purposes.

## Experiments

| File | Question it answers | Metric |
|---|---|---|
| `hellaswag_evaluation.py` | Does the model still pick the correct HellaSwag ending at this dtype? | accuracy, per-candidate NLL |
| `fineweb_evaluation.py` | How much does held-out FineWeb10B validation loss change at this dtype? | token-level NLL (EOT-masked; document boundaries are not attention-isolated, matching how the model was pretrained) |

Both scripts warm up with one throwaway batch (the same batch reused as the first
real one, so shapes/memory allocation are already warmed up) before timing starts,
then append one row per run to a shared, per-dataset CSV, recording `dtype`,
`elapsed_time_sec`, `avg_NLL`, `accuracy` (HellaSwag only), etc. Each run's raw
per-sample NLLs are also saved to `output/`, linked to
its CSV row via `run_id`.

`data/` (HellaSwag val set, FineWeb10B shard) is gitignored, same as the other
experiment folders — regenerate it via `Data_download.ipynb` + `Data_process.ipynb`.

## Analyzing results

Load either CSV with pandas and group by `dtype` to compare accuracy/NLL/speed
across the sweep. For a closer look at one specific run (e.g. per-sample NLL
distribution), load its `output/*.pt` file with `torch.load` using the `run_id`
from the CSV row of interest.

## Running this (Google Colab)

Like the other experiment folders in this repository, this is meant to be run on
Google Colab (GPU runtime):

1. Upload this `Quantization/` folder to Google Drive (e.g. under
   `MyDrive/Project/nanoGPT/Quantization/`).
2. Run `Data_download.ipynb` then `Data_process.ipynb` to populate `data/`.
3. Run `hellaswag_evaluation.py` / `fineweb_evaluation.py` directly (adjust the
   `dtype` entry in each script's `config` dict per run).

Setup steps may still change while this experiment is in progress.
