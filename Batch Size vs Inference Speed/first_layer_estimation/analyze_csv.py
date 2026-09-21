"""
Analyzes the CSV produced by first_layer_estimation.py.

Groups rows by batch_size (averaging across --repeats), then reports two
different "speed" comparisons against batch_size=1:

1. Total time (fair, same-workload basis): since --target_tokens fixes the
   total number of tokens processed to the same value for every batch_size,
   comparing total_time_ms directly answers "how much faster/slower is this
   batch_size at getting the same amount of work done". Ranking by this
   column is the "which batch_size is fastest overall" ranking.
2. Per-call time: pure_forward_mean_ms naturally grows with batch_size (a
   bigger batch takes longer per forward call - that alone isn't
   interesting). What IS interesting is comparing that growth against what
   pure linear scaling would predict (call_time_ratio_vs_B1 vs
   linear_expectation=batch_size/1) - efficiency_vs_linear=1.0 means exactly
   linear, >1.0 means worse than linear (super-linear degradation), <1.0
   means better than linear.

Usage:
    python analyze_csv.py --csv output/ascending/fineweb_block0_nsight_evaluation.csv

    # also average ascending+descending together (pools both files' rows,
    # so the average is weighted by each file's repeat count):
    python analyze_csv.py --csv output/ascending/fineweb_block0_nsight_evaluation.csv \\
        --csv2 output/descending/fineweb_block0_nsight_evaluation.csv
"""

import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=str, required=True)
parser.add_argument('--csv2', type=str, default=None,
                     help="Optional second CSV - typically the other sweep direction "
                          "(point --csv at output/ascending/... and this at "
                          "output/descending/..., or vice versa). When given, also "
                          "prints a third report that pools both files' rows together "
                          "and averages per batch_size across both directions.")
args = parser.parse_args()

REQUIRED_COLS = {'batch_size', 'repeat_idx', 'total_time_ms', 'pure_forward_mean_ms'}


def load(path):
    df = pd.read_csv(path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise SystemExit(f"{path} is missing expected column(s): {sorted(missing)}. "
                          f"Is this really a first_layer_estimation.py output CSV?")
    return df


def direction_label(df, path):
    if 'reversed_batch' not in df.columns:
        return path
    vals = df['reversed_batch'].unique()
    if len(vals) > 1:
        return f"{path} (MIXED ascending+descending rows)"
    return f"{path} ({'descending' if vals[0] else 'ascending'})"


def analyze(df, label):
    if 'reversed_batch' in df.columns and df['reversed_batch'].nunique() > 1:
        print("WARNING: this pools ascending and descending rows together - if that's not "
              "intentional (e.g. via --csv2), session-order effects are mixed into the "
              "batch_size comparison below.")
        print()

    agg_kwargs = dict(
        total_time_ms_mean=('total_time_ms', 'mean'),
        total_time_ms_std=('total_time_ms', 'std'),
        pure_forward_mean_ms_mean=('pure_forward_mean_ms', 'mean'),
        pure_forward_mean_ms_std=('pure_forward_mean_ms', 'std'),
        n_repeats=('repeat_idx', 'count'),
    )
    if 'per_token_us' in df.columns:
        agg_kwargs['per_token_us_mean'] = ('per_token_us', 'mean')
    if 'qkv_proj_mean_ms' in df.columns:
        agg_kwargs['qkv_proj_mean_ms_mean'] = ('qkv_proj_mean_ms', 'mean')
        agg_kwargs['qkv_proj_mean_ms_std'] = ('qkv_proj_mean_ms', 'std')

    grouped = df.groupby('batch_size').agg(**agg_kwargs).reset_index().sort_values('batch_size')
    grouped['total_time_ms_std'] = grouped['total_time_ms_std'].fillna(0.0)
    grouped['pure_forward_mean_ms_std'] = grouped['pure_forward_mean_ms_std'].fillna(0.0)
    if 'qkv_proj_mean_ms_std' in grouped.columns:
        grouped['qkv_proj_mean_ms_std'] = grouped['qkv_proj_mean_ms_std'].fillna(0.0)

    if 1 not in grouped['batch_size'].values:
        raise SystemExit(f"{label}: batch_size=1 isn't present in this data - can't compute "
                          f"speedup relative to B=1.")

    base_total = grouped.loc[grouped['batch_size'] == 1, 'total_time_ms_mean'].iloc[0]
    base_call = grouped.loc[grouped['batch_size'] == 1, 'pure_forward_mean_ms_mean'].iloc[0]

    grouped['speedup_total_time_vs_B1'] = base_total / grouped['total_time_ms_mean']  # >1 = faster than B=1 at the same total workload, <1 = slower
    grouped['call_time_ratio_vs_B1'] = grouped['pure_forward_mean_ms_mean'] / base_call  # how many times as long one forward call takes, vs B=1's call
    grouped['linear_expectation'] = grouped['batch_size'] / 1  # what call_time_ratio_vs_B1 "should" be if scaling were perfectly linear
    grouped['efficiency_vs_linear'] = grouped['call_time_ratio_vs_B1'] / grouped['linear_expectation']  # 1.0 = exactly linear, >1.0 = worse (super-linear), <1.0 = better

    if 'qkv_proj_mean_ms_mean' in grouped.columns:
        base_qkv = grouped.loc[grouped['batch_size'] == 1, 'qkv_proj_mean_ms_mean'].iloc[0]
        grouped['qkv_call_time_ratio_vs_B1'] = grouped['qkv_proj_mean_ms_mean'] / base_qkv
        grouped['qkv_efficiency_vs_linear'] = grouped['qkv_call_time_ratio_vs_B1'] / grouped['linear_expectation']

    # Raw pure_forward_mean_ms trivially grows with batch_size (one call now does more work), so
    # ranking by it directly is close to meaningless - it's ~always ordered by batch_size. Instead,
    # normalize each batch_size's per-call time down onto a per-B=1-sample basis (divide by B) so
    # the kernel-only ranking is a fair, same-workload comparison like the total-time one - and so
    # it's directly comparable to whole_inference_estimation.py's normalized_* columns and the
    # blog write-up's tables, which use the same B=1 basis. This CSV has no precomputed normalized
    # column, so it's computed here instead.
    grouped['normalized_pure_forward_mean_ms'] = grouped['pure_forward_mean_ms_mean'] / grouped['batch_size']
    grouped['normalized_pure_forward_std_ms'] = grouped['pure_forward_mean_ms_std'] / grouped['batch_size']
    base_normalized = grouped.loc[grouped['batch_size'] == 1, 'normalized_pure_forward_mean_ms'].iloc[0]
    grouped['speedup_normalized_vs_B1'] = base_normalized / grouped['normalized_pure_forward_mean_ms']

    if 'qkv_proj_mean_ms_mean' in grouped.columns:
        grouped['normalized_qkv_proj_mean_ms'] = grouped['qkv_proj_mean_ms_mean'] / grouped['batch_size']
        grouped['normalized_qkv_proj_std_ms'] = grouped['qkv_proj_mean_ms_std'] / grouped['batch_size']
        base_qkv_normalized = grouped.loc[grouped['batch_size'] == 1, 'normalized_qkv_proj_mean_ms'].iloc[0]
        grouped['speedup_qkv_normalized_vs_B1'] = base_qkv_normalized / grouped['normalized_qkv_proj_mean_ms']

    ranked_total = grouped.sort_values('total_time_ms_mean')  # fastest (smallest total time to process the same target_tokens) first
    ranked_call = grouped.sort_values('normalized_pure_forward_mean_ms')  # fastest (smallest normalized pure forward/kernel time) first

    print(f"=== {label} ===")
    print(f"Loaded {len(df)} rows")
    print(f"Batch sizes found: {sorted(int(b) for b in grouped['batch_size'])}")
    print()
    print("Ranked fastest -> slowest, by total time to process the same amount of tokens (target_tokens):")
    for _, row in ranked_total.iterrows():
        B = int(row['batch_size'])
        line = (f"  batch_size={B:<3d}"
                f" | total time: {row['total_time_ms_mean']:>10,.3f}±{row['total_time_ms_std']:>8,.3f} ms"
                f" ({row['speedup_total_time_vs_B1']:.3f}x vs B=1)"
                f" | per-call: {row['pure_forward_mean_ms_mean']:>10,.3f}±{row['pure_forward_mean_ms_std']:>7,.3f} ms/call"
                f" ({row['call_time_ratio_vs_B1']:.3f}x vs B=1, linear would be {row['linear_expectation']:.0f}x,"
                f" efficiency vs linear={row['efficiency_vs_linear']:.3f})")
        if 'per_token_us_mean' in row:
            line += f" | per-token: {row['per_token_us_mean']:.4f} us/token"
        if 'qkv_proj_mean_ms_mean' in row:
            line += (f" | QKV_PROJ only: {row['qkv_proj_mean_ms_mean']:>8,.3f}±{row['qkv_proj_mean_ms_std']:>6,.3f} ms/call"
                      f" (efficiency vs linear={row['qkv_efficiency_vs_linear']:.3f})")
        line += f" | n_repeats={int(row['n_repeats'])}"
        print(line)

    print()
    print(f"Ranked fastest -> slowest, by pure forward (kernel, block0 forward only) time, "
          f"normalized to a per-B=1-sample basis (divide by B):")
    for _, row in ranked_call.iterrows():
        B = int(row['batch_size'])
        line = (f"  batch_size={B:<3d}"
                f" | normalized: {row['normalized_pure_forward_mean_ms']:>10,.3f}±{row['normalized_pure_forward_std_ms']:>7,.3f} ms"
                f" (/{B})"
                f" ({row['speedup_normalized_vs_B1']:.3f}x vs B=1)"
                f" | raw per-call: {row['pure_forward_mean_ms_mean']:>10,.3f}±{row['pure_forward_mean_ms_std']:>7,.3f} ms/call"
                f" ({row['call_time_ratio_vs_B1']:.3f}x vs B=1, linear would be {row['linear_expectation']:.0f}x,"
                f" efficiency vs linear={row['efficiency_vs_linear']:.3f})"
                f" | total time: {row['total_time_ms_mean']:>10,.3f}±{row['total_time_ms_std']:>8,.3f} ms"
                f" ({row['speedup_total_time_vs_B1']:.3f}x vs B=1)")
        if 'per_token_us_mean' in row:
            line += f" | per-token: {row['per_token_us_mean']:.4f} us/token"
        if 'normalized_qkv_proj_mean_ms' in row:
            line += (f" | QKV_PROJ normalized: {row['normalized_qkv_proj_mean_ms']:>8,.3f}±{row['normalized_qkv_proj_std_ms']:>6,.3f} ms"
                      f" ({row['speedup_qkv_normalized_vs_B1']:.3f}x vs B=1)")
        line += f" | n_repeats={int(row['n_repeats'])}"
        print(line)


df1 = load(args.csv)
analyze(df1, direction_label(df1, args.csv))

if args.csv2:
    print()
    df2 = load(args.csv2)
    analyze(df2, direction_label(df2, args.csv2))

    print()
    combined = pd.concat([df1, df2], ignore_index=True)
    analyze(combined, "COMBINED average (pools rows from both files above)")
