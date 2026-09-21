# Memory measurement

GPU memory profiling of nanoGPT's GPT-2 — capturing PyTorch's CUDA caching-allocator
snapshots to see where memory actually goes on a single forward+loss pass over the
very first validation batch, at two levels of detail: the whole model, and a single
self-attention layer in isolation.

This is part of the broader [AI efficiency study](../README.md) in this repository
and is written up in more detail in the accompanying blog post.

## What was changed vs. original nanoGPT

Each experiment's entry-point script started from [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)
`train.py`, then stripped down to a single forward+loss pass instead of an actual
training run — no training loop, no optimizer step, no checkpointing/wandb. These
scripts exist purely to produce one memory snapshot; they never train the model.

- `dtype` is applied directly to the model weights via `model.to(device, dtype=...)`
  rather than through `torch.autocast`, so the model's parameters and activations
  are computed in that dtype end-to-end.
- `model.py`'s `CausalSelfAttention` always uses the manual/"math" attention
  implementation (no `scaled_dot_product_attention` fast-path), so the same kernel
  path is used regardless of `dtype`, keeping snapshots directly comparable across
  dtype settings.
- `print_gpu_memory(tag)` — a helper that prints `torch.cuda.memory_allocated` /
  `memory_reserved` / `max_memory_allocated` / `max_memory_reserved` (current and
  peak, in GB) at various tagged points.
- Around the very first `'val'`-split batch inside `estimate_loss()` (`k == 0`),
  the forward(+loss) call is wrapped in `torch.cuda.reset_peak_memory_stats()` →
  `torch.cuda.memory._record_memory_history(max_entries=100000)` → the call →
  `torch.cuda.memory._dump_snapshot(...)` → `_record_memory_history(enabled=None)`,
  writing a full CUDA allocator history to a `.pickle` in `out/`.
  - `First_Batch_Memory.py` wraps that call directly, so the snapshot covers the
    **whole model's** forward + loss.
  - `SelfAttention_Memory.py` instead registers a forward pre/post hook on just
    `model.transformer.h[0].attn` — block 0's self-attention submodule — so only
    *that one submodule's* forward is captured, isolating its memory footprint
    from the rest of the model.

## Experiments

| Folder | Question it answers | Entry point |
|---|---|---|
| `First_batch/` | How is GPU memory allocated across the *whole model's* forward + loss on the very first validation batch? | `First_Batch_Measurement.ipynb` → `First_Batch_Memory.py --batch_size=12 --compile=False` → `out/first_batch_memory.pickle` |
| `One_selfattention/` | Of that memory, how much is attributable to just one self-attention layer (block 0), in isolation? | `SelfAttention_Memory_Mesurement.ipynb` → `SelfAttention_Memory.py --batch_size=12 --compile=False` → `out/attn_snapshot.pickle` |

Both scripts take the usual nanoGPT command-line overrides via `configurator.py`
(e.g. `--batch_size=`, `--compile=`, `--dtype=`).

The evaluation code in the `.ipynb` notebooks was written with Gemini.

## Running this (Google Colab)

These experiments were run on Google Colab (GPU runtime), not locally:

1. Prepare nanoGPT's OpenWebText data (`train.bin`, `val.bin`, `meta.pkl`) via
   nanoGPT's own `data/openwebtext/prepare.py` — not included here — and place it
   at `Memory measurement/data/openwebtext/` (gitignored; `../data/openwebtext`
   relative to each experiment folder).
2. Upload this `Memory measurement/` folder (with `data/` alongside it) to Google
   Drive, e.g. under `MyDrive/Project/nanoGPT/Memory measurement/`.
3. Open the notebook for the experiment you want to run.
4. In the `%cd` cell, update the path to match where you actually placed the
   folder in your Drive (the committed notebooks still point at older path names
   from before this folder was reorganized).
5. Run all cells top to bottom. The last cell shells out to that folder's
   `*_Memory.py` entry-point script.

## Analyzing results

Each run writes one `.pickle` snapshot to `out/`. Upload it to
[pytorch.org/memory_viz](https://pytorch.org/memory_viz) (or use
`torch.cuda._memory_viz`) to see the allocator's timeline — every allocation/free,
grouped by call stack, over the course of that one forward pass.

## Environment

Captured from a run on Google Colab:

```
OS: Linux-6.6.122+-x86_64-with-glibc2.35
Python: 3.12.13 (main, Mar  4 2026, 09:23:07) [GCC 11.4.0]
PyTorch: 2.11.0+cu128
CUDA (torch built with): 12.8
cuDNN version: 91900
GPU available: True
GPU name: Tesla T4
GPU compute capability: (7, 5)
GPU total memory (GB): 15.637086208
transformers: 5.15.0
Driver Version: 580.82.07   CUDA Version (driver): 13.0
```
