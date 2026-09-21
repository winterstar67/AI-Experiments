"""
Sweeps the block0-only forward (GPTQ-style: StopIteration right after block0)
over a fixed set of batch sizes, timing each with cuda.Event.

--reversed_batch controls the sweep order (ascending 1->64 by default). This
also doubles as the overheating-hypothesis check that used to live in a
separate 3_reversed_batch_order.py: if a batch's timing depends on whether it
runs early or late in the sweep (not just on its own size), running with
--reversed_batch and comparing against the ascending run reveals that - a
batch that's relatively worse only when it runs late in EITHER direction
points to session-progression effects (e.g. GPU heat/clock) rather than
something intrinsic to that batch_size. Kept as one script instead of two
separate files so the two directions can't drift out of sync with each other.
"""

import torch
import torch.nn as nn
import os
from model import GPT, QKV_PROJ_EVENTS
import traceback
import numpy as np
import argparse
import statistics
import subprocess
import warnings
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore", message=".*Profiler clears events.*")  # cosmetic only - we create a fresh profiler per batch_size and don't need events kept across cycles

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]  # fixed sweep set - order is controlled by --reversed_batch, not by a CLI list, so the two directions can't diverge

def gpu_status():
    # For the overheating/clock hypothesis check: query temperature (C), SM clock (MHz), and power (W) on demand
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=temperature.gpu,clocks.sm,power.draw',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        ).stdout.strip()
        temp, clock, power = [x.strip() for x in out.split(',')]
        return {"temp_c": float(temp), "sm_clock_mhz": float(clock), "power_w": float(power)}
    except Exception as e:
        return {"temp_c": None, "sm_clock_mhz": None, "power_w": None, "error": str(e)}

def format_gpu_transition(before, after):
    # Compact "before -> after" for one print line instead of two separate before/after lines
    if before.get("temp_c") is None or after.get("temp_c") is None:
        return "(nvidia-smi query failed)"
    return (f"{before['temp_c']:.0f}→{after['temp_c']:.0f}C, "
            f"{before['sm_clock_mhz']:.0f}→{after['sm_clock_mhz']:.0f}MHz, "
            f"{before['power_w']:.2f}→{after['power_w']:.2f}W")

DATA_DIR = "../../Quantization/data"
FINEWEB_10B = DATA_DIR + "/fineweb10B"
DATASET_NAME = "fineweb"

def load_fineweb():
    path = FINEWEB_10B+"/fineweb_val_000000.bin"
    data = np.fromfile(path, dtype=np.uint16, offset=256*4)
    return data

# -----------------------------------------------------------------------------
init_from = 'gpt2'

def str2bool(v):
    if v.lower() in ('true', '1', 'yes'):
        return True
    elif v.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got: {v!r}")

parser = argparse.ArgumentParser()
parser.add_argument('--dtype', type=str, required=True,
                     choices=['float64', 'float32', 'bfloat16', 'float16', 'float8_e4m3fn', 'float8_e5m2', 'int8'],
                     help="Model/computation dtype to run the sweep in.")
parser.add_argument('--reversed_batch', type=str2bool, default=False,
                     help="false = ascending sweep 1->64 (old first_layer_estimation.py), "
                          "true = descending sweep 64->1 (old 3_reversed_batch_order.py) - see module docstring. "
                          "Only controls OUT_DIR/the CSV's reversed_batch column when used with --only_batch, "
                          "since a single batch_size has no order of its own.")
parser.add_argument('--target_tokens', type=int, default=256*1024,
                     help="Fixes the total number of tokens processed in the measured window regardless of "
                          "batch_size - iters_for_B is derived automatically as target_tokens // (B*T).")
parser.add_argument('--repeats', type=int, default=5,
                     help="Repeat the whole warmup+timed measurement this many times per batch_size, for a "
                          "more precise timing estimate (mean±std across repeats, not just across the "
                          "iterations within one repeat).")
parser.add_argument('--skip_perfetto', action='store_true',
                     help="Turn this on when using ncu/nsys together - if torch.profiler grabs CUPTI first, "
                          "ncu/nsys can't see the kernels (CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED).")
parser.add_argument('--only_batch', type=int, default=None,
                     help="Run just this one batch_size instead of sweeping BATCH_SIZES - for ncu/nsys "
                          "recipes (e.g. 4_ncu_bound_investigation) that need to isolate a single batch so "
                          "--launch-skip can target its exact last iteration without other batch sizes' "
                          "iterations mixed into the count.")
parser.add_argument('--no_save', action='store_true',
                     help="Skip writing to csv_path - for throwaway runs (e.g. a --only_batch warm-up "
                          "before the real ascending/descending sweep) that shouldn't pollute the real "
                          "results CSV. Only skips the CSV write, which already happens after all "
                          "cuda.Event timing is done, so this has zero effect on the measurement itself.")
parser.add_argument('--skip_gpu_status', action='store_true',
                     help="Suppress the 'GPU (before run -> after run)' console print - for runs wrapped "
                          "by ncu, where that print is misleading: ncu's own kernel-replay overhead "
                          "(--set full does dozens of replay passes with memory save/restore per launch) "
                          "serializes execution enough that clock/temp/power never reach what an "
                          "unwrapped run reaches, so the numbers don't reflect real GPU behavior.")
parser.add_argument('--csv_name', type=str, default=None,
                     help="Override the output CSV's filename (still saved under OUT_DIR, i.e. "
                          "./output/ascending or ./output/descending). Default: "
                          "'{DATASET_NAME}_block0_nsight_evaluation.csv'. Useful for keeping a distinct "
                          "experiment condition (e.g. a clock-locked run) in its own file instead of "
                          "overwriting or mixing rows into the default sweep's CSV.")
args = parser.parse_args()

# Separate output subfolder per sweep direction, so ascending and descending
# runs (which otherwise touch the exact same batch_sizes) never overwrite or
# mix results together.
OUT_DIR = os.path.join("./output", "descending" if args.reversed_batch else "ascending")
os.makedirs(OUT_DIR, exist_ok=True)

config = {
    "token_size": 1024,
    "warmup": 10,  # Re-warm up this many times independently for each batch_size (so batch order doesn't affect results)
    "seed": 1337,
    "device": "cuda",
    "dtype": args.dtype,
    "torch.backends.cuda.matmul.allow_tf32": False,
    "torch.backends.cudnn.allow_tf32": False,
    "torch.use_deterministic_algorithms": True
    }

seed = config['seed']
T = config['token_size']
warmup = config['warmup']
dtype = config['dtype']
device = config['device']
cuda_allow_tf32 = config['torch.backends.cuda.matmul.allow_tf32']
cudnn_allow_tf32 = config['torch.backends.cudnn.allow_tf32']
deterministic_kernel = config['torch.use_deterministic_algorithms']

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = cuda_allow_tf32
torch.backends.cudnn.allow_tf32 = cudnn_allow_tf32
torch.use_deterministic_algorithms(deterministic_kernel)

ptdtype = {
    'float64': torch.float64,
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
    'float16': torch.float16,
    'float8_e4m3fn': torch.float8_e4m3fn,
    'float8_e5m2': torch.float8_e5m2,
    'int8': torch.int8
    }[dtype]

# === Load the model exactly once (all batch_sizes share this load/CUDA context init cost) ===
model = GPT.from_pretrained(init_from, dict(dropout=0.0))
model.to(device=device, dtype=ptdtype)
model.eval()

# === Same pattern as GPTQ: force-stop with StopIteration right after block0's forward ===
def stop_hook(module, input, output):
    raise StopIteration
block0_handle = model.transformer.h[0].register_forward_hook(stop_hook)

if args.only_batch is not None:
    batch_sizes = [args.only_batch]
else:
    batch_sizes = list(reversed(BATCH_SIZES)) if args.reversed_batch else list(BATCH_SIZES)

# Sized generously so data won't run out through the largest batch_size's warmup + the target_tokens-sized timed window
max_B = max(batch_sizes)
VAL_TOKENS = warmup * max_B * T + args.target_tokens + T * 2
fineweb_val = load_fineweb()[:VAL_TOKENS]
print(f"VAL_TOKENS={VAL_TOKENS:,} (max_batch={max_B}, target_tokens={args.target_tokens:,})")

csv_path = os.path.join(OUT_DIR, args.csv_name or f"{DATASET_NAME}_block0_nsight_evaluation.csv")

# A full sweep (no --only_batch) is meant to be self-contained - re-running it shouldn't require
# manually deduping the CSV against whatever leftover rows (past runs, --only_batch throwaways
# that forgot --no_save) happened to already be sitting in it. --only_batch runs still append
# (the normal case is they pass --no_save anyway), so this only fires for the real, full sweep.
if not args.no_save and args.only_batch is None and os.path.exists(csv_path):
    os.remove(csv_path)

try:
    print("Dtype:", ptdtype)
    print(f"Batch sizes to sweep ({'descending' if args.reversed_batch else 'ascending'}):", batch_sizes)
    print("Repeats per batch_size:", args.repeats)

    for B in batch_sizes:
        BT = B*T
        iters_for_B = max(1, args.target_tokens // BT)  # How many iterations this batch_size needs to reach target_tokens
        print(f"\n=== batch_size={B} (iters={iters_for_B:,}, tokens to process={iters_for_B*BT:,}) ===")

        repeat_total_tokens_processed = []
        repeat_total_time_ms = []
        repeat_pure_forward_mean_ms = []
        repeat_qkv_proj_mean_ms = []

        for repeat_idx in range(args.repeats):
            torch.manual_seed(seed)  # Same seed every repeat too - the computation is deterministic, so only timing should vary run to run

            gpu_before = gpu_status()

            with torch.no_grad():
                # ======= Warmup stage (redone fresh each repeat) =======
                samples = torch.tensor(fineweb_val[:(BT+1)].astype(np.int64))
                x, y = samples[:-1].view(B, -1).to(device), samples[1:].view(B, -1).to(device)
                for _ in range(warmup):
                    try:
                        _ = model(x, y)
                    except StopIteration:
                        pass
                torch.cuda.synchronize()
                # ======= Warmup stage =======

                # === Start recording a trace viewable in Perfetto UI (ui.perfetto.dev) ===
                # Only on the last repeat: by then the GPU/caches have settled from whatever the
                # earlier repeats warmed up, so this trace should be more representative than the
                # first repeat's. Profiler CPU-side instrumentation overhead can delay kernel
                # launches and inflate the cuda.Event timing we're trying to measure, and creating a
                # fresh with_stack=True profiler on every repeat piles up host RAM, so we still only
                # do this once per batch_size rather than on all `--repeats` runs.
                prof = None
                if not args.skip_perfetto and repeat_idx == args.repeats - 1:
                    prof = torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                        record_shapes=True,
                        with_stack=True,
                    )
                    prof.start()
                forward_events = []  # [for logging] pure time measured with cuda.Event, covering only the model() call (through block0)
                QKV_PROJ_EVENTS.clear()  # warmup-phase QKV_PROJ events (if any) don't belong in this repeat's timing
                for idx in range(iters_for_B):
                    torch.cuda.nvtx.range_push("DATA_PREP")
                    offset = (warmup + idx) * (BT + 1)  # offset so this doesn't overlap the portion consumed by warmup
                    samples = fineweb_val[offset:offset+(BT+1)]
                    _B = (len(samples)-1)//T
                    if _B <= 0:
                        torch.cuda.nvtx.range_pop()
                        break
                    samples = torch.tensor(samples[:(_B*T+1)].astype(np.int64))
                    x, y = samples[:-1].view(_B, -1).to(device), samples[1:].view(_B, -1).to(device)
                    torch.cuda.synchronize()
                    torch.cuda.nvtx.range_pop()

                    torch.cuda.nvtx.range_push("MODEL_FORWARD_BLOCK0")
                    start_evt = torch.cuda.Event(enable_timing=True)
                    end_evt = torch.cuda.Event(enable_timing=True)
                    try:
                        start_evt.record()
                        _ = model(x, y)
                        end_evt.record()
                    except StopIteration:
                        end_evt.record()
                    torch.cuda.nvtx.range_pop()
                    forward_events.append((start_evt, end_evt))
                    if prof is not None:
                        prof.step()

                torch.cuda.synchronize()
                if prof is not None:
                    prof.stop()

            gpu_after = gpu_status()

            pure_forward_times_ms = [s.elapsed_time(e) for s, e in forward_events]
            mean_ms = statistics.mean(pure_forward_times_ms) if pure_forward_times_ms else 0.0
            std_ms = statistics.stdev(pure_forward_times_ms) if len(pure_forward_times_ms) > 1 else 0.0
            total_ms = sum(pure_forward_times_ms)
            total_tokens_processed = len(pure_forward_times_ms) * BT
            per_token_us = (total_ms / total_tokens_processed * 1000) if total_tokens_processed else 0.0

            # Real (non-ncu) single-kernel timing for just the QKV_PROJ matmul - directly comparable
            # to ncu's isolated QKV_PROJ Duration, but without ncu's 31-pass replay overhead.
            qkv_proj_times_ms = [s.elapsed_time(e) for s, e in QKV_PROJ_EVENTS]
            qkv_mean_ms = statistics.mean(qkv_proj_times_ms) if qkv_proj_times_ms else 0.0
            qkv_std_ms = statistics.stdev(qkv_proj_times_ms) if len(qkv_proj_times_ms) > 1 else 0.0

            print(f"  [repeat {repeat_idx+1}/{args.repeats}]")
            print(f"    Tokens processed:                            {total_tokens_processed:,}")
            print(f"    Total elapsed (sum of {len(pure_forward_times_ms)} calls):          {total_ms:,.3f} ms")
            print(f"    BLOCK0 Forward (kernel, model forward only): {mean_ms:,.3f}±{std_ms:,.3f} ms/call")
            print(f"    QKV_PROJ only (kernel, real/non-ncu):        {qkv_mean_ms:,.3f}±{qkv_std_ms:,.3f} ms/call")
            print(f"    Time per token:                              {per_token_us:.4f} us/token")
            if not args.skip_gpu_status:
                print(f"    GPU (before run → after run):                {format_gpu_transition(gpu_before, gpu_after)}")

            # === Save the Perfetto trace: open with ui.perfetto.dev or chrome://tracing ===
            if prof is not None:
                # No "_reversed" filename suffix needed - OUT_DIR already separates ascending/descending
                perfetto_path = os.path.join(OUT_DIR, f"{DATASET_NAME}_perfetto_block0_{dtype}_b{B}.json")
                prof.export_chrome_trace(perfetto_path)
                print(f"    Perfetto trace saved:                        {perfetto_path}")

            repeat_total_tokens_processed.append(total_tokens_processed)
            repeat_total_time_ms.append(total_ms)
            repeat_pure_forward_mean_ms.append(mean_ms)
            repeat_qkv_proj_mean_ms.append(qkv_mean_ms)

            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M")

            # === CSV: per-dataset run ledger, appended across runs (one row per batch_size x repeat) ===
            if not args.no_save:
                if os.path.exists(csv_path):
                    results_df = pd.read_csv(csv_path)
                else:
                    results_df = pd.DataFrame()

                new_row = pd.DataFrame([{
                    "batch_size": B,
                    "repeat_idx": repeat_idx,
                    "reversed_batch": args.reversed_batch,
                    "datetime": timestamp,
                    "dataset": DATASET_NAME,
                    "dtype": dtype,
                    "token_size": T,
                    "target_tokens": args.target_tokens,
                    "iters": len(pure_forward_times_ms),
                    "total_tokens_processed": total_tokens_processed,
                    "total_time_ms": total_ms,  # sum of pure per-call forward times (cuda.Event), through block0 only
                    "pure_forward_mean_ms": mean_ms,
                    "pure_forward_std_ms": std_ms,
                    "qkv_proj_mean_ms": qkv_mean_ms,  # real (non-ncu) single-kernel QKV_PROJ timing - see QKV_PROJ_EVENTS in model.py
                    "qkv_proj_std_ms": qkv_std_ms,
                    "per_token_us": per_token_us,
                    "gpu_temp_before_c": gpu_before.get("temp_c"),
                    "gpu_sm_clock_before_mhz": gpu_before.get("sm_clock_mhz"),
                    "gpu_power_before_w": gpu_before.get("power_w"),
                    "gpu_temp_after_c": gpu_after.get("temp_c"),
                    "gpu_sm_clock_after_mhz": gpu_after.get("sm_clock_mhz"),
                    "gpu_power_after_w": gpu_after.get("power_w"),
                }])
                results_df = pd.concat([results_df, new_row], ignore_index=True)
                results_df.to_csv(csv_path, index=False)

            # Free this repeat's tensors before the next repeat/batch_size - the caching allocator
            # can otherwise fragment across differently-sized allocations within this single process.
            del x, y, samples, forward_events
            torch.cuda.empty_cache()

        # === Summary across the `--repeats` runs for this batch_size ===
        total_time_mean = statistics.mean(repeat_total_time_ms)
        total_time_std = statistics.stdev(repeat_total_time_ms) if len(repeat_total_time_ms) > 1 else 0.0
        pure_forward_mean_of_means = statistics.mean(repeat_pure_forward_mean_ms)
        pure_forward_std_of_means = statistics.stdev(repeat_pure_forward_mean_ms) if len(repeat_pure_forward_mean_ms) > 1 else 0.0
        qkv_proj_mean_of_means = statistics.mean(repeat_qkv_proj_mean_ms)
        qkv_proj_std_of_means = statistics.stdev(repeat_qkv_proj_mean_ms) if len(repeat_qkv_proj_mean_ms) > 1 else 0.0

        print()
        print(f"[batch_size={B}] over {args.repeats} repeats:")
        print(f"    Tokens processed:                            {repeat_total_tokens_processed[0]:,}")
        print(f"    Total elapsed:                                {total_time_mean:,.3f}±{total_time_std:,.3f} ms")
        print(f"    BLOCK0 Forward (kernel, model forward only): {pure_forward_mean_of_means:,.3f}±{pure_forward_std_of_means:,.3f} ms/call")
        print(f"    QKV_PROJ only (kernel, real/non-ncu):        {qkv_proj_mean_of_means:,.3f}±{qkv_proj_std_of_means:,.3f} ms/call")

    block0_handle.remove()

except Exception as e:
    traceback.print_exc()
