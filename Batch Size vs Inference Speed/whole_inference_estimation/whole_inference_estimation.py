import torch
import os
from model import GPT
import time
import traceback
import pandas as pd
from datetime import datetime
import numpy as np
import argparse
import statistics

DATA_DIR = "../../Quantization/data"
FINEWEB_10B = DATA_DIR + "/fineweb10B"
DATASET_NAME = "fineweb"

def load_fineweb():
    path = FINEWEB_10B+"/fineweb_val_000000.bin"
    data = np.fromfile(path, dtype=np.uint16, offset=256*4)  # Load the whole file into memory instead of np.memmap (same as fineweb_evaluation.py)
    return data

# -----------------------------------------------------------------------------
init_from = 'gpt2'

BATCH_SIZES = [1, 2, 4, 8, 16]  # fixed sweep set - capped at 16 (unlike first_layer_estimation.py, which goes up to 64) because this script runs the full forward through lm_head and computes softmax/log-sum-exp over the full (B, T, vocab=50257) logits for NLL - that logits tensor alone is already several GB at B=32+ and OOMs on a ~15GB GPU. first_layer_estimation.py stops right after block0 and never reaches that memory-heavy softmax step, so it can go past 16. Order is controlled by --reversed_batch, not by this list, so the two directions can't diverge.

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
                     help="false = ascending sweep 1->16, true = descending sweep 16->1. Only "
                          "controls OUT_DIR/the CSV's reversed_batch column when used with "
                          "--only_batch, since a single batch_size has no order of its own.")
parser.add_argument('--target_tokens', type=int, default=256*1024,
                     help="Fixes the total number of tokens processed in the measured window regardless of "
                          "batch_size - iters_for_B is derived automatically as target_tokens // (B*T).")
parser.add_argument('--repeats', type=int, default=5,
                     help="Repeat the whole warmup+timed measurement this many times per batch_size, for a "
                          "more precise timing estimate (mean±std across repeats, not just across the "
                          "iterations within one repeat).")
parser.add_argument('--only_batch', type=int, default=None,
                     help="Run just this one batch_size instead of sweeping BATCH_SIZES - e.g. for a "
                          "standalone GPU warm-up run (batch_size=1) before the real ascending/descending sweep.")
parser.add_argument('--no_save', action='store_true',
                     help="Skip writing to csv_path and the per-repeat .pt artifact - for throwaway runs "
                          "(e.g. a --only_batch warm-up) that shouldn't pollute the real results/output "
                          "folder. Only skips disk writes, which already happen after all cuda.Event timing "
                          "is done, so this has zero effect on the measurement itself.")
parser.add_argument('--csv_name', type=str, default=None,
                     help="Override the output CSV's filename (still saved under OUT_DIR, i.e. "
                          "./output/ascending or ./output/descending). Default: "
                          "'{DATASET_NAME}_nsight_evaluation.csv'. Useful for keeping a distinct experiment "
                          "condition (e.g. a clock-locked run) in its own file instead of overwriting or "
                          "mixing rows into the default sweep's CSV.")
args = parser.parse_args()

# Separate output subfolder per sweep direction, so ascending and descending
# runs (which otherwise touch the exact same batch_sizes) never overwrite or
# mix results together in the same CSV.
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

# No GPTQ hook here - the full forward pass runs through all 12 blocks + lm_head (this is what differs from scripts 2/3)

if args.only_batch is not None:
    batch_sizes = [args.only_batch]
else:
    batch_sizes = list(reversed(BATCH_SIZES)) if args.reversed_batch else list(BATCH_SIZES)

# Sized generously so data won't run out through the largest batch_size's warmup + the target_tokens-sized timed window.
# Deliberately max(BATCH_SIZES) rather than max(batch_sizes): with --only_batch, batch_sizes is a single-element
# list, so max(batch_sizes) would just be whatever one batch_size was requested.
max_B = max(BATCH_SIZES)
VAL_TOKENS = warmup * max_B * T + args.target_tokens + T * 2
fineweb_val = load_fineweb()[:VAL_TOKENS]
print(f"VAL_TOKENS={VAL_TOKENS:,} (max_batch={max_B}, target_tokens={args.target_tokens:,})")

csv_path = os.path.join(OUT_DIR, args.csv_name or f"{DATASET_NAME}_nsight_evaluation.csv")

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
        print(f"\n=== batch_size={B} (iters={iters_for_B:,}, tokens to process={iters_for_B*BT:,}, normalize /{B}) ===")

        repeat_total_tokens_processed = []
        repeat_total_time_ms = []       # sum of pure per-call forward times (cuda.Event), per repeat - comparable to 2_/3_'s "total elapsed"
        repeat_elapsed_sec = []         # full CPU wall-clock time (includes data prep + NLL calc), per repeat
        repeat_pure_forward_mean_ms = []
        repeat_avg_NLL = []

        for repeat_idx in range(args.repeats):
            torch.manual_seed(seed)  # Same seed every repeat too - the computation is deterministic, so only timing should vary run to run

            with torch.no_grad():
                # ======= Warmup stage (redone fresh each repeat) =======
                samples = torch.tensor(fineweb_val[:(BT+1)].astype(np.int64))
                x, y = samples[:-1].view(B, -1).to(device), samples[1:].view(B, -1).to(device)
                for _ in range(warmup):
                    logits = model(x, y)
                torch.cuda.synchronize()
                # ======= Warmup stage =======

                torch.cuda.reset_peak_memory_stats()

                fineweb_NLLs = []
                forward_events = []  # [for logging] pure time measured with cuda.Event, covering only the model() call (data prep/transfer excluded)
                data_prep_times_ms = []  # [for logging] CPU wall-clock (ms) for the DATA_PREP section - includes the H2D transfer (up to the sync point)

                # No torch.profiler here on purpose: the goal is a precise pure_forward mean±std
                # from cuda.Event timing alone, and the profiler's CPU-side instrumentation overhead
                # can itself delay kernel launches and inflate the very time we're trying to measure
                # (on top of being what exhausted host RAM when it ran every repeat). For kernel-level
                # inspection of this script, use the nsys/ncu cells in Float32.ipynb instead.
                CPU_time_start = time.perf_counter()
                for idx in range(iters_for_B):
                    torch.cuda.nvtx.range_push("DATA_PREP")
                    data_prep_start = time.perf_counter()
                    try:
                        offset = idx * (BT + 1)
                        samples = fineweb_val[offset:offset+(BT+1)]
                        _B = (len(samples)-1)//T
                        if _B <= 0:
                            break
                        samples = torch.tensor(samples[:(_B*T+1)].astype(np.int64))
                        x, y = samples[:-1].view(_B, -1).to(device), samples[1:].view(_B, -1).to(device)
                        masking = (y != 50256).to(device)
                        torch.cuda.synchronize()  # [for logging] the H2D transfer is async, so without this the data prep time would be undermeasured
                    finally:
                        data_prep_times_ms.append((time.perf_counter() - data_prep_start) * 1000)
                        torch.cuda.nvtx.range_pop()

                    torch.cuda.nvtx.range_push("MODEL_FORWARD")
                    start_evt = torch.cuda.Event(enable_timing=True)
                    end_evt = torch.cuda.Event(enable_timing=True)
                    try:
                        start_evt.record()
                        logits = model(x, y) # [batch_size, Token_size, vocab_size] - probabilities
                        end_evt.record()
                    finally:
                        torch.cuda.nvtx.range_pop()
                    forward_events.append((start_evt, end_evt))

                    logits_f = logits.float() # low-precision dtypes cause log(0.0)=-inf errors, so upcast to float for headroom
                    m = torch.max(logits_f, dim=-1, keepdim=True).values # max used to avoid exp() overflow
                    log_Z = m.squeeze(-1) + torch.log(torch.sum(torch.exp(logits_f - m), dim=-1)) # log-sum-exp
                    label_logits = torch.gather(logits_f, dim=-1, index=y.unsqueeze(-1)).squeeze(-1)
                    label_log_probabilities = label_logits - log_Z # log(softmax(x)) = x - logsumexp(x)
                    NLL = -1*torch.sum(label_log_probabilities*masking, dim=-1)/masking.sum(dim=-1)

                    fineweb_NLLs.extend(NLL.tolist())

                torch.cuda.synchronize()
                CPU_time_end = time.perf_counter()

            peak_memory_allocated_gb = torch.cuda.max_memory_allocated() / (1024**3)
            peak_memory_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)

            elapsed_time = CPU_time_end - CPU_time_start
            avg_NLL = sum(fineweb_NLLs) / len(fineweb_NLLs) if fineweb_NLLs else float('nan')
            pure_forward_times_ms = [s.elapsed_time(e) for s, e in forward_events]  # [for logging] the pure time of only the model() call, per idx
            pure_forward_mean_ms = statistics.mean(pure_forward_times_ms) if pure_forward_times_ms else 0.0
            pure_forward_std_ms = statistics.stdev(pure_forward_times_ms) if len(pure_forward_times_ms) > 1 else 0.0
            data_prep_mean_ms = statistics.mean(data_prep_times_ms) if data_prep_times_ms else 0.0  # [for logging]
            data_prep_std_ms = statistics.stdev(data_prep_times_ms) if len(data_prep_times_ms) > 1 else 0.0  # [for logging]
            total_tokens_processed = len(pure_forward_times_ms) * BT
            total_time_ms = sum(pure_forward_times_ms)  # sum of pure per-call forward times - comparable to first_layer_estimation.py's "total elapsed"
            normalized_mean_ms = pure_forward_mean_ms / B  # ms/call scaled down to a per-B=1-sample basis - if scaling were perfectly linear, this would be equal across all batch_sizes
            normalized_std_ms = pure_forward_std_ms / B

            print(f"  [repeat {repeat_idx+1}/{args.repeats}]")
            print(f"    Tokens processed:                          {total_tokens_processed:,}")
            print(f"    Elapsed (wall time, incl. data prep + NLL): {elapsed_time*1000:,.3f} ms")
            print(f"    Total Forward-only (sum of {len(pure_forward_times_ms)} calls):    {total_time_ms:,.3f} ms")
            print(f"    Pure Forward (kernel, model forward only): {pure_forward_mean_ms:,.3f}±{pure_forward_std_ms:,.3f} ms/call")
            print(f"    Normalized (/{B}):                          {normalized_mean_ms:,.3f}±{normalized_std_ms:,.3f} ms/call")
            print(f"    Average NLL:                                {avg_NLL:.4f}")

            repeat_total_tokens_processed.append(total_tokens_processed)
            repeat_total_time_ms.append(total_time_ms)
            repeat_elapsed_sec.append(elapsed_time)
            repeat_pure_forward_mean_ms.append(pure_forward_mean_ms)
            repeat_avg_NLL.append(avg_NLL)

            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M")
            file_timestamp = now.strftime("%Y%m%d-%H%M")

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
                    "iters": len(fineweb_NLLs),
                    "num_samples": len(fineweb_NLLs),
                    "total_tokens_processed": total_tokens_processed,
                    "total_time_ms": total_time_ms,  # sum of pure per-call forward times (cuda.Event) - matches the "total time" printed above
                    "elapsed_time_sec": elapsed_time,  # full CPU wall-clock time (data prep + forward + NLL calc)
                    "pure_forward_mean_ms": pure_forward_mean_ms,  # [for logging] mean time (ms) of only the model() call, measured with cuda.Event - excludes data prep/transfer
                    "pure_forward_std_ms": pure_forward_std_ms,  # [for logging]
                    "normalized_pure_forward_mean_ms": normalized_mean_ms,  # pure_forward_mean_ms / B - comparable across batch_sizes on a per-B=1-sample basis
                    "normalized_pure_forward_std_ms": normalized_std_ms,
                    "data_prep_mean_ms": data_prep_mean_ms,  # [for logging] mean CPU wall-clock time (ms) for DATA_PREP (slicing + tensor creation + H2D transfer)
                    "data_prep_std_ms": data_prep_std_ms,  # [for logging]
                    "avg_NLL": avg_NLL,
                    "peak_memory_allocated_gb": peak_memory_allocated_gb,
                    "peak_memory_reserved_gb": peak_memory_reserved_gb,
                }])
                results_df = pd.concat([results_df, new_row], ignore_index=True)
                results_df.to_csv(csv_path, index=False)

                # === torch.save: raw per-run artifact, linked to the CSV row via batch_size + repeat_idx + datetime ===
                pt_path = os.path.join(OUT_DIR, f"{DATASET_NAME}_nsight_{dtype}_b{B}_r{repeat_idx}_{file_timestamp}.pt")
                torch.save({
                    "dtype": dtype,
                    "dataset": DATASET_NAME,
                    "batch_size": B,
                    "repeat_idx": repeat_idx,
                    "iters": len(fineweb_NLLs),
                    "fineweb_NLLs": fineweb_NLLs,
                    "pure_forward_times_ms": pure_forward_times_ms,  # [for logging] raw per-idx values
                    "data_prep_times_ms": data_prep_times_ms,  # [for logging] raw per-idx values
                }, pt_path)

            # Free this repeat's tensors (logits over the full vocab is the big one, up to several GB)
            # before the next repeat/batch_size - otherwise the caching allocator can fragment across
            # differently-sized allocations within this single process and OOM earlier than a fresh
            # process would (this script sweeps all batch_sizes x repeats in one process).
            del logits, x, y, samples, forward_events
            torch.cuda.empty_cache()

        # === Summary across the `--repeats` runs for this batch_size ===
        elapsed_mean_ms = statistics.mean(repeat_elapsed_sec) * 1000
        elapsed_std_ms = (statistics.stdev(repeat_elapsed_sec) if len(repeat_elapsed_sec) > 1 else 0.0) * 1000
        total_time_mean = statistics.mean(repeat_total_time_ms)
        total_time_std = statistics.stdev(repeat_total_time_ms) if len(repeat_total_time_ms) > 1 else 0.0
        pure_forward_mean_of_means = statistics.mean(repeat_pure_forward_mean_ms)
        pure_forward_std_of_means = statistics.stdev(repeat_pure_forward_mean_ms) if len(repeat_pure_forward_mean_ms) > 1 else 0.0
        normalized_mean_of_means = pure_forward_mean_of_means / B
        normalized_std_of_means = pure_forward_std_of_means / B
        avg_NLL_mean = statistics.mean(repeat_avg_NLL)

        print()
        print(f"[batch_size={B}] over {args.repeats} repeats:")
        print(f"    Tokens processed:                          {repeat_total_tokens_processed[0]:,}")
        print(f"    Elapsed (wall time):                       {elapsed_mean_ms:,.3f}±{elapsed_std_ms:,.3f} ms")
        print(f"    Total Forward-only:                        {total_time_mean:,.3f}±{total_time_std:,.3f} ms")
        print(f"    Pure Forward (kernel, model forward only): {pure_forward_mean_of_means:,.3f}±{pure_forward_std_of_means:,.3f} ms/call")
        print(f"    Normalized (/{B}):                          {normalized_mean_of_means:,.3f}±{normalized_std_of_means:,.3f} ms/call")
        print(f"    Average NLL:                                {avg_NLL_mean:.4f}")

except Exception as e:
    traceback.print_exc()
