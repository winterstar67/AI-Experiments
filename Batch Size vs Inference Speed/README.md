# Batch Size vs Inference Speed

Investigates whether `batch_size` actually affects nanoGPT's GPT-2 *pure forward*
inference speed on a single GPU. This was motivated by an unexpected finding while
building GPTQ quantization ([`Quantization/GPTQ/1_Quantization/GPTQ_Quantization.py`](../Quantization/README.md)):
`batch_size=1` and `batch_size=64` showed no meaningful difference in per-sample
forward time. This project isolates that question and re-measures it directly (no
GPTQ hooks, no calibration), at two levels of detail — the whole model, and just
the first transformer block, matching GPTQ's own calibration hook boundary.

This is part of the broader [AI efficiency study](../README.md) in this repository
and is written up in more detail in the accompanying blog post
(`A_larger_batch_doesnt_always_increase_speed`).

## Finding

Larger `batch_size` doesn't reliably speed up inference on a T4 GPU. **Root
cause: GPU power-cap clock throttling** (T4 has a 70W power budget, no external
power connector), not kernel/compute inefficiency — the kernel itself actually
gets *more* efficient with batch_size (`ncu` Compute (SM) Throughput 84%→95.6%
from `batch_size=1`→`64`, via shrinking tail-effect waste, not occupancy, which
stays flat at 50%).

Confirmed with a controlled experiment: locking the GPU clock with
`nvidia-smi -lgc 585` (T4's boost clock is 1590 MHz) turns "batch_size makes it
*worse* than linear scaling" (free/default clock, up to 1.3x worse) into
"batch_size makes it *better* than linear scaling" (locked clock, down to ~0.78x
of linear) — reproduced at three scopes: the whole model, block 0 alone, and the
QKV_PROJ kernel in isolation. Per-`batch_size` clock-lock telemetry also confirms
the 585 MHz lock held with zero exceptions across every `batch_size` tested (so
the comparison isn't confounded by uneven lock quality), and that average power
draw itself increases monotonically with `batch_size` — the same reason a small
`batch_size` can sustain a higher boost clock under the *default* (unlocked)
clock policy: it simply draws less power, leaving more headroom under the 70W
cap.

Along the way, resolved a separate measurement-methodology puzzle: `ncu`,
`torch.profiler`/Perfetto UI, and a direct `cuda.Event` wrap of the same kernel
each report a different "kernel time" for the *same* kernel. That's not a bug —
`ncu` locks the clock to ~585 MHz for reproducibility (so it disagrees with a
free-clock measurement even for the exact same kernel), and a direct
`cuda.Event` wrap includes CPU→GPU dispatch/launch latency that a profiler's own
kernel-only duration excludes. Don't compare absolute "kernel time" numbers
across tools without accounting for both.

## What was changed vs. original nanoGPT

`model.py` is [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)
`model.py`, used unmodified via `GPT.from_pretrained('gpt2')` — this investigation
doesn't touch the model implementation, aside from one small addition:
`CausalSelfAttention.forward` wraps `self.c_attn(x)` (the QKV_PROJ matmul) in its
own `cuda.Event` pair (`QKV_PROJ_EVENTS`), so `first_layer_estimation.py` can time
that one kernel in isolation during a real (non-`ncu`) run — directly comparable
in scope to the `ncu` QKV_PROJ drill-down, without `ncu`'s clock lock or replay
overhead in the way. The other structural addition is in
`first_layer_estimation.py`, which registers a forward hook on `transformer.h[0]`
that raises `StopIteration` right after block 0 finishes — the same
force-stop-after-block0 trick `GPTQ_Quantization.py`'s calibration pass uses — so
timing is isolated to exactly the portion GPTQ's calibration actually runs.

The actual *execution* code — loading the model, constructing the input batch,
the block0 force-stop hook — is carried over from `GPTQ_Quantization.py`. The
*measurement* code around it (the `cuda.Event` timing harness, `--repeats`/
`--reversed_batch`/`--only_batch` sweep logic, GPU status/warm-up logging,
CSV/`.pt` output, `analyze_csv.py`, and the `ncu`/power-check notebook cells)
was written with Claude.

Both scripts time only the `model()` call itself via `torch.cuda.Event`, with
**no `torch.profiler` wrapping the timed loop** — an earlier version of this
methodology (traced back to `Kernel investigation/`, then reused in `KV Cache/`
and `Quantization/GPTQ/0_Performance_Baseline/`) always ran a profiler around the
measurement loop, which inflates and destabilizes the very timing being measured.
`first_layer_estimation.py` still supports an optional, deliberately-isolated
Perfetto trace (`torch.profiler`, one export per `batch_size`, only on the last
`--repeats` iteration) for kernel-level inspection — never during the actual
timed measurement of other repeats.

## Experiments

| Folder | Question it answers | Entry point |
|---|---|---|
| `whole_inference_estimation/` | Does `batch_size` affect the *whole model's* (all 12 blocks + `lm_head` + NLL) pure forward speed? | `whole_inference_estimation.ipynb` → `whole_inference_estimation.py --dtype float32 [--reversed_batch {false,true}]` |
| `first_layer_estimation/` | Same question, isolated to just block 0 (GPTQ's calibration boundary) — and if batch_size *does* matter, which kernel and why (`ncu` `SpeedOfLight`/`--set full` drill-down, plus a direct QKV_PROJ `cuda.Event` measurement)? | `first_layer_estimation.ipynb` → `first_layer_estimation.py --dtype float32 [--reversed_batch {false,true}]` |

Both scripts sweep a fixed `BATCH_SIZES` list once per process (loading the model
only once), timing each `batch_size` for `--repeats` independent
warmup+measurement cycles, at a fixed `--target_tokens` total (so every
`batch_size` processes the same amount of data — `iters` is derived as
`target_tokens // (batch_size * token_size)`). `whole_inference_estimation` caps
out at `batch_size=16` (the full-vocab softmax logits tensor for NLL gets
multi-GB beyond that on a ~15GB GPU); `first_layer_estimation` goes up to `64`
since it stops before `lm_head`.

- **`--reversed_batch`** (`true`/`false`, defaults to `false`) sweeps
  `BATCH_SIZES` ascending or descending, saving to a separate
  `output/ascending/` or `output/descending/` subfolder. Running both
  directions and comparing is itself an experiment — it's how a suspected GPU
  overheating/session-order confound (a `batch_size` appearing slower only
  because it ran *late* in a long sweep, not because of its own size) gets
  checked: if the same `batch_size` times differently depending on sweep
  direction, that points to session-order effects rather than something
  intrinsic to that batch size. Only matters for the full sweep — with
  `--only_batch`, a single batch_size has no order of its own, so it just
  controls which output subfolder that run's CSV row lands in.
- **`--only_batch <N>`** runs just one `batch_size` instead of the full sweep
  — used for `ncu` recipes that need to isolate a single batch's last
  iteration, for per-`batch_size` clock/power telemetry logging (loop over
  `--only_batch` once per `batch_size` with its own `nvidia-smi` log file), and
  for a throwaway GPU warm-up run before the real sweep. A full sweep (no
  `--only_batch`) deletes any pre-existing CSV at start so re-running it stays
  self-contained; `--only_batch` runs append instead, so a `--csv_name`-driven
  loop over batch_sizes accumulates into one CSV.
- **`--no_save`** skips the CSV/`.pt`/(`whole_inference_estimation` also skips) disk writes for a
  throwaway run (e.g. a `--only_batch 1` warm-up) — it only skips writes that
  already happen *after* all `cuda.Event` timing is recorded, so it has no effect
  on the measurement itself.
- **`--csv_name <name>`** overrides the output CSV's filename (still saved
  under `output/ascending/` or `output/descending/`) — useful for keeping a
  distinct experiment condition (e.g. a clock-locked run) in its own file
  instead of mixing rows into the default sweep's CSV.

## Analyzing results

Each folder has its own `analyze_csv.py --csv <path> [--csv2 <other_direction_path>]`
(e.g. `output/ascending/fineweb_nsight_evaluation.csv`). It groups rows by
`batch_size` (averaging across `--repeats`) and prints two rankings:

1. **Total time to process the same `target_tokens`** — directly comparable
   across `batch_size` since the workload is fixed, so this ranks which
   `batch_size` is fastest overall, with the fold change vs. `batch_size=1`.
2. **Pure forward (kernel) time, normalized** onto a per-`batch_size=1`-sample
   basis (`pure_forward_mean_ms / batch_size`) — raw per-call time trivially
   grows with `batch_size`, so this normalization is what makes the per-call
   ranking meaningful, alongside an "efficiency vs. linear" score (1.0 =
   exactly linear scaling, <1.0 = better than linear). `first_layer_estimation`'s
   version also reports this for the `qkv_proj_mean_ms` column when present
   (i.e. when the CSV came from a run with `QKV_PROJ_EVENTS` timing enabled).

Pass `--csv2` to also average a second CSV (typically the other sweep
direction) together with the first — prints a third "combined" report pooling
both files' rows, weighted by each file's repeat count.

## Clock-locked experiments

`first_layer_estimation`'s notebook (and `whole_inference_estimation`'s companion
`whole_inference_estimation_fixing_clock.ipynb`)
include cells that lock the GPU clock with `nvidia-smi -lgc <freq>` (`-rgc` to
release it afterward), then re-run the sweep — this is what isolates the
clock-throttling effect from everything else. A background `nvidia-smi
--query-gpu=... -lms 200` logger runs alongside each measurement to verify the
lock actually held (`-lgc` is a request, not a guarantee — it can still get
pulled down by the power cap under heavy load) and to check
`clocks_event_reasons.sw_power_cap`/`sw_thermal_slowdown` directly, rather than
inferring throttling from power/clock correlation alone. Looping `--only_batch`
per `batch_size` with its own log file (instead of one log for a whole
multi-batch sweep) gives a per-`batch_size` breakdown of lock quality, not just
an aggregate.

## Running this (Google Colab)

These experiments were run on Google Colab (GPU runtime), not locally:

1. Upload the contents of this folder (`whole_inference_estimation/`,
   `first_layer_estimation/`, etc.) to Google Drive under
   `MyDrive/Project/nanoGPT/Batch Size vs Inference Speed/` — the notebooks'
   `%cd` cells expect `whole_inference_estimation/` and
   `first_layer_estimation/` to sit directly under that path.
2. Make sure `Quantization/data/fineweb10B/` (this project reuses that data —
   see [`Quantization/README.md`](../Quantization/README.md) for how to
   regenerate it) is present at the relative path each script expects
   (`../../Quantization/data`).
3. Open the notebook for the experiment you want to run
   (`whole_inference_estimation.ipynb` or `first_layer_estimation.ipynb`).
4. In each `%cd` cell, update the path to match where you actually placed the
   folder in your Drive.
5. Run cells top to bottom: environment check → `nsys` PATH setup → GPU warm-up
   → ascending sweep → descending sweep → (`first_layer_estimation` only) `ncu` drill-down / power
   check → clock-locked re-runs.

## Environment

Captured from a run of `whole_inference_estimation.ipynb` on Google Colab:

```
OS: Linux-6.6.122+-x86_64-with-glibc2.39
Python: 3.13.15 (main, Aug  6 2026, 11:06:22) [GCC 13.3.0]
PyTorch: 2.11.0+cu128
CUDA (torch built with): 12.8
cuDNN version: 91900
GPU available: True
GPU name: Tesla T4
GPU compute capability: (7, 5)
GPU total memory (GB): 15.637086208
transformers: 5.16.1
Driver Version: 580.82.07   CUDA Version (driver): 13.0
```
