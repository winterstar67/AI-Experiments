# Kernel investigation

Investigates whether CUDA kernel selection for nanoGPT's GPT-2 forward pass is
sensitive to tensor dimension alignment — vocab size, batch size, and token
(sequence) length/padding — by capturing `torch.profiler` traces across different
combinations and comparing which kernels actually get dispatched.

This was originally motivated by replicating [Andrej Karpathy's nanoGPT vocab_size
padding](https://github.com/karpathy/nanoGPT) optimization (50257 → 50304, the
nearest multiple of 64), to understand *why* padding an odd dimension up to a nicer
number changes kernel selection at all.

This is part of the broader [AI efficiency study](../README.md) in this repository
and is written up in more detail in the accompanying blog post.

## What was changed vs. original nanoGPT

`model.py` is [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)
`model.py`, used unmodified — this investigation doesn't touch the model
implementation itself. Instead, each experiment script varies how the *input* is
constructed (batch size, sequence padding scheme, dtype/autocast) and wraps the
forward pass in a `torch.profiler` context to record which CUDA kernels get
launched for each combination.

## Experiments

| File | Question it answers | Model | vocab_size |
|---|---|---|---|
| `Data_download.ipynb` | Downloads the HellaSwag and FineWeb data used by the experiments below | — | — |
| `Data_process.ipynb` | Tokenizes/prepares the downloaded HellaSwag data for evaluation | — | — |
| `vocab_padding_test.py` → `vocab_padding_case.ipynb` | Does kernel selection differ across dtype (fp32/fp16) × batch_size (25/26) × padding scheme (odd/even, 16-aligned), with the padded vocab size? | randomly-initialized `GPT(config)` | 50304 (padded, nanoGPT default) |
| `No_vocab_padding_test.py` → `No_vocab_padding_case.ipynb` | Same sweep, but with the original (unpadded) GPT-2 vocab size | `GPT.from_pretrained('gpt2')` | 50257 (forced by `from_pretrained`) |

Both test scripts construct the model **once**, outside the dtype/batch_size/padding
sweep — since none of those swept variables change the model's weights (autocast
casts ops on the fly and keeps fp32 master weights; batch size and padding only
affect input construction) — and reuse it across every combination for a fair,
apples-to-apples comparison. Each combination's `torch.profiler` chrome trace is
exported to `output/`, named by dtype, batch size, and padding/input shape, for
inspection in the Perfetto UI (`https://ui.perfetto.dev`).

`data/hellaswag (Quantization)/` and `data/fineweb10B/` hold the datasets themselves
(gitignored — not committed, for copyright reasons; regenerate locally via
`Data_download.ipynb` + `Data_process.ipynb`).

The evaluation code in the `.ipynb` notebooks was written with Gemini.

## Running this (Google Colab)

These experiments were run on Google Colab (GPU runtime), not locally:

1. Upload this `Kernel investigation/` folder to Google Drive (e.g. under
   `MyDrive/Project/nanoGPT/Kernel investigation/`).
2. Run `Data_download.ipynb` then `Data_process.ipynb` to populate `data/`.
3. Open the notebook for the experiment you want to run
   (`vocab_padding_case.ipynb` or `No_vocab_padding_case.ipynb`).
4. In the `%cd` cell, update the path to match where you actually placed the
   folder in your Drive.
5. Run all cells top to bottom.

Each notebook `%cd`s into `Evaluation/` and shells out to that folder's
`*_test.py`, so the working directory must match before running.

## Analyzing results

Each run exports one Chrome trace JSON per combination into `output/`. These can
be uploaded directly to the [Perfetto UI](https://ui.perfetto.dev) to inspect the
actual CUDA kernel launched for each op (name, duration, shapes) and compare
kernel selection across combinations.

## Environment

Captured from a run of `vocab_padding_case.ipynb` on Google Colab:

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
transformers: 5.13.1
Driver Version: 580.82.07   CUDA Version (driver): 13.0
```
