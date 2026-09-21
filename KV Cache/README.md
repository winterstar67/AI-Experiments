# KV Cache

Implementation and evaluation of KV caching for nanoGPT's GPT-2 model — built from
scratch to understand *why* caching key/value activations speeds up autoregressive
generation, then verified for correctness and measured for latency at increasing
levels of detail (end-to-end → per-token → per-self-attention-layer → per-kernel).

This is part of the broader [AI efficiency study](../README.md) in this repository
and is written up in more detail in the accompanying blog post.

## What was changed vs. original nanoGPT

`model.py` is [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) `model.py`,
edited to add a KV cache to `CausalSelfAttention`:

- `CausalSelfAttention.save_KV` / `load_KV` / `reset_KV` — store and reuse each layer's
  key/value tensors across generation steps instead of recomputing them from the full
  sequence every step.
- `save_KV_activation` / `load_KV_activation` — a parallel, cache-free logging path used
  only to verify correctness (dumps the "ground truth" K/V a non-cached forward pass
  would produce, for comparison against the cached ones).
- `forward()` on `CausalSelfAttention`, `Block`, and `GPT` all take a `running_mode`
  argument (`"train"`, `"inf_no_cache"`, `"inf_cache"`) that switches between normal
  training-style forward, full-recompute inference, and cached inference.

Every experiment folder below has its own copy of `model.py` and `configurator.py`
(unmodified nanoGPT config-override helper) so each Colab notebook is self-contained
and runnable on its own.

## Experiments

| Folder | Question it answers | Entry point |
|---|---|---|
| `0_KV_Cache_verification/` | Does the cache actually produce the same K/V values as a full recompute? Checked at both `float32` and `float64` precision. | `KV_cache_verification.ipynb` → `fp32_test_file.py`, `fp64_test_file.py` |
| `1_end_to_end_time_estimation/` | How much wall-clock time does caching save over a full generation run? | `end_to_end_time_estimation.ipynb` → `end_to_end_estimation_test_file.py` |
| `2_time_estimation_per_token/` | How does per-token latency scale with sequence length, with vs. without cache? | `Time_estimation_per_token.ipynb` → `one_token_estimation_test_file.py`, plots in `Graph drawing along the token length/` (`KV_cache_case.py` / `No_KV_cache_case.py`) |
| `3_time_estimation_of_one_SelfAttention_Layer/` | How much of the per-token time difference is attributable to the self-attention layer specifically? | `KVCache_selfattention_estimation.ipynb` → `one_selfattention_layer_estimation.py` (uses `torch.profiler`, exports a Chrome trace per token step) |
| `4_Nsight_analysis/` | Inside that self-attention layer, why is the cached QKᵀ matmul so much faster than the recomputed one — down to the GPU kernel and its throughput/occupancy? | `Nsight_analysis.ipynb` → `KV_cache_Nsight_analysis_test_file.py` / `No_KV_cache_Nsight_analysis_test_file.py` (uses `ncu`/Nsight Compute) |

**0 → correctness first, 1 → 4 → progressively finer-grained timing.** Run them in
that order if reproducing from scratch, since later stages assume the cache is
already known to be correct.

The evaluation/graph-plotting code in the `.ipynb` notebooks was written with Gemini.

### 4_Nsight_analysis in more detail

`model.py` here adds a `layer_idx` to `CausalSelfAttention`/`Block` and wraps
*only* the raw QKᵀ matmul — `att = (q @ k.transpose(-2, -1))`, not the `1/√d_k`
scaling that follows it — in `torch.cuda.nvtx.range_push/pop("QK_MATMUL")`, and
only when `layer_idx == 11` (the last layer) and `T == 1023` (context fully
filled), i.e. exactly the single matmul call examined in
`3_time_estimation_of_one_SelfAttention_Layer/`. The notebook then runs Nsight
Compute (`ncu`) with `--nvtx-include "QK_MATMUL/"` so it profiles only that one
scoped kernel launch instead of the whole script, against both the cached and the
recomputed matmul for a direct comparison:

- `--section SpeedOfLight`, cached case (`KV_cache_...`) → `KV_cache_case_SpeedOfLight.ncu-rep`
- `--section SpeedOfLight`, recomputed case (`No_KV_cache_...`) → `No_KV_cache_case_SpeedOfLight.ncu-rep`
- `--section MemoryWorkloadAnalysis --section WarpStateStats`, cached case → `KV_cache_Memory_Warp.ncu-rep`
- `--set full` (the complete metric set, e.g. for arithmetic-intensity/roofline
  analysis), cached case only → `KV_cache_Arithmentic_intensity.ncu-rep`

`.ncu-rep` files aren't committed (regenerate them by running the notebook) — open
them in the Nsight Compute UI (`ncu-ui`) or via `ncu --import <file> --page details`,
as the notebook's own follow-up cells do. This step needs `ncu` on `PATH` in
whatever environment runs the notebook, which is not preinstalled on stock Colab
GPU runtimes.

## Running this (Google Colab)

These experiments were run on Google Colab (GPU runtime), not locally:

1. Upload this `KV Cache/` folder to Google Drive (e.g. under
   `MyDrive/Project/nanoGPT/KV Cache/`).
2. Open the notebook for the experiment you want to run.
3. In the first cell (`%cd "/content/drive/MyDrive/Project/nanoGPT/KV Cache/..."`),
   update the path to match where you actually placed the folder in your Drive.
4. Run all cells top to bottom.

Each notebook `%cd`s into its own experiment folder and shells out to that folder's
entry-point script (see the table above), so the working directory must match before
running.
