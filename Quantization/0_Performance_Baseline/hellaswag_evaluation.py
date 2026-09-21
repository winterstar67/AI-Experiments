import torch
import os
from contextlib import nullcontext
from model import GPT
import time
import json
import traceback
import gc
import pandas as pd
import uuid
from datetime import datetime
import argparse

DATA_DIR = "../data"
OUT_DIR = "./output"
os.makedirs(OUT_DIR, exist_ok=True)

HELLASWAG_DIR = DATA_DIR + "/hellaswag"
DATASET_NAME = "hellaswag"

def load_hellaswag():
    path = HELLASWAG_DIR+"/tokenized_hellaswag_val.json"
    with open(path, "r") as f:
        examples = json.load(f)
    return examples

# -----------------------------------------------------------------------------
init_from = 'gpt2'

parser = argparse.ArgumentParser()
parser.add_argument('--dtype', type=str, required=True,
                     choices=['float64', 'float32', 'bfloat16', 'float16', 'float8_e4m3fn', 'float8_e5m2', 'int8'])
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--use_flash', action='store_true')  # If True, use F.scaled_dot_product_attention (Flash/SDPA); otherwise fall back to manual attention as before
args = parser.parse_args()

config = {
    "experiment_repeat": 1, # Will be used on time estimation
    "batch_size": args.batch_size,
    "warmup": 10,
    "seed": 1337,
    "device": "cuda",
    "dtype": args.dtype,
    "torch.backends.cuda.matmul.allow_tf32": False,
    "torch.backends.cudnn.allow_tf32": False,
    "torch.use_deterministic_algorithms": True
    }

seed = config['seed']
dtype = config['dtype']
batch_size = config['batch_size']
warmup = config['warmup']
device = config['device']
cuda_allow_tf32 = config['torch.backends.cuda.matmul.allow_tf32']
cudnn_allow_tf32 = config['torch.backends.cudnn.allow_tf32']
deterministic_kernel = config['torch.use_deterministic_algorithms']

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = cuda_allow_tf32
torch.backends.cudnn.allow_tf32 = cudnn_allow_tf32
torch.use_deterministic_algorithms(deterministic_kernel)
device_type = 'cuda' if 'cuda' in device else 'cpu'

ptdtype = {
    'float64': torch.float64,
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
    'float16': torch.float16,
    'float8_e4m3fn': torch.float8_e4m3fn,
    'float8_e5m2': torch.float8_e5m2,
    'int8': torch.int8
    }[dtype]

model = GPT.from_pretrained(init_from, dict(dropout=0.0, use_flash=args.use_flash))
model.to(device=device, dtype=ptdtype)

hella_swag_val = load_hellaswag()
# === Sorting to prevent segmentation OOM
hella_swag_val = sorted(
    hella_swag_val,
    key=lambda x: max([len(c) for c in x['candidates']]),
    reverse=True
)

# === descending order check - Reason of ordering: preventing run from OOM by GPU memory segmentation
prev = 1e+5
curr = 0
for x in hella_swag_val:
    curr = max([len(c) for c in x['candidates']])
    if prev < curr:
        print(curr)
        break
    else:
        prev = curr

# Checking whether any sample has more than 4 candidates
for sample in hella_swag_val:
    if len(sample['candidates']) != 4:
        print(sample['candidates'])
prev = None # Memory free

print("= = = = = samples")
print("samples:", hella_swag_val[0])

try:
    print("Dtype:", ptdtype)
    torch.manual_seed(seed)

    model.eval()
    # ======= Warmup sample preparation
    samples = hella_swag_val[:batch_size]
    samples_len = len(samples)
    candidates = [i for x in samples for i in x['candidates']]
    masking_info = [i for x in samples for i in x['masking_info']]
    tokens_len = [len(x) for x in candidates]
    max_token_len = max(tokens_len)

    x = torch.zeros((samples_len*4, max_token_len),dtype=torch.int64)
    padded_masking = torch.zeros((samples_len*4, max_token_len-1),dtype=torch.int32)
    for _idx in range(samples_len*4):
        x[_idx, :tokens_len[_idx]] = torch.tensor(candidates[_idx])
        padded_masking[_idx, :tokens_len[_idx]-1] = torch.tensor(masking_info[_idx])[1:]
    padded_masking = padded_masking.to(device)
    x = x.to(device)
    y = x[:, 1:].contiguous()
    x = x[:, :-1].contiguous()
    # ======= Warmup sample preparation

    num_iter = len(hella_swag_val)//batch_size
    accuracy_check, hellaswag_NLLs, all_candidate_NLLs = [], [], []
    print_interval = max(num_iter//10,1)

    with torch.no_grad():
        # ======= Warmup stage
        for _ in range(warmup):
            logits = model(x,y)
        # ======= Warmup stage
        
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        CPU_time_start = time.perf_counter()
        for idx in range(num_iter+1):
            if (idx==num_iter) and (len(hella_swag_val)%batch_size==0):
                break
            samples = hella_swag_val[idx*batch_size:(idx+1)*batch_size]
            samples_len = len(samples)
            candidates = [i for x in samples for i in x['candidates']]
            masking_info = [i for x in samples for i in x['masking_info']]
            tokens_len = [len(x) for x in candidates]
            max_token_len = max(tokens_len)

            x = torch.zeros((samples_len*4, max_token_len),dtype=torch.int64)
            padded_masking = torch.zeros((samples_len*4, max_token_len-1),dtype=torch.int32)
            for _idx in range(samples_len*4):
                x[_idx, :tokens_len[_idx]] = torch.tensor(candidates[_idx])
                padded_masking[_idx, :tokens_len[_idx]-1] = torch.tensor(masking_info[_idx])[1:]
            padded_masking = padded_masking.to(device)
            x = x.to(device)
            y = x[:, 1:].contiguous()
            x = x[:, :-1].contiguous()
            logits = model(x, y) # [batch_size, Token_size, vocab_size] - probabilities
            logits_f = logits.float() # low-precision dtypes can cause log(0.0)=-inf errors, so upcast to float for precision
            m = torch.max(logits_f, dim=-1, keepdim=True).values # max value to prevent exp() overflow
            log_Z = m.squeeze(-1) + torch.log(torch.sum(torch.exp(logits_f - m), dim=-1)) # log-sum-exp
            label_logits = torch.gather(logits_f, dim=-1, index=y.unsqueeze(-1)).squeeze(-1)
            label_log_probabilities = label_logits - log_Z # log(softmax(x)) = x - logsumexp(x)
            NLL = -1*torch.sum(label_log_probabilities*padded_masking, dim=-1)/padded_masking.sum(dim=-1)

            # Grouping
            for _idx in range(samples_len):
                sample_NLLs = NLL[_idx*4:(_idx+1)*4]
                predicted_label = torch.argmin(sample_NLLs)
                sample_NLL = sample_NLLs[samples[_idx]['label']]
                if predicted_label==samples[_idx]['label']:
                    accuracy_check.append(1)
                else:
                    accuracy_check.append(0)
                hellaswag_NLLs.append(sample_NLL.item())
                all_candidate_NLLs.append(sample_NLLs.tolist())
            if idx%print_interval == 0:
                print(f"{round(100*idx/num_iter, 1)}% Done - Average accuracy:", sum(accuracy_check)/len(accuracy_check))
        torch.cuda.synchronize()
        print(f"100% Done - Average accuracy:", sum(accuracy_check)/len(accuracy_check))
        CPU_time_end = time.perf_counter()
        peak_memory_allocated_gb = torch.cuda.max_memory_allocated() / (1024**3)
        peak_memory_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)

        elapsed_time = CPU_time_end - CPU_time_start
        avg_NLL = sum(hellaswag_NLLs) / len(hellaswag_NLLs)
        accuracy = sum(accuracy_check) / len(accuracy_check)
        run_id = uuid.uuid4().hex[:8]
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M")
        file_timestamp = now.strftime("%Y%m%d-%H%M")

        # === CSV: per-dataset run ledger, appended across runs ===
        csv_path = os.path.join(OUT_DIR, f"{DATASET_NAME}_evaluation.csv")
        if os.path.exists(csv_path):
            results_df = pd.read_csv(csv_path)
        else:
            results_df = pd.DataFrame()

        new_row = pd.DataFrame([{
            "run_id": run_id,
            "datetime": timestamp,
            "dataset": DATASET_NAME,
            "dtype": dtype,
            "batch_size": batch_size,
            "num_samples": len(accuracy_check),
            "elapsed_time_sec": elapsed_time,
            "avg_NLL": avg_NLL,
            "accuracy": accuracy,
            "peak_memory_allocated_gb": peak_memory_allocated_gb,
            "peak_memory_reserved_gb": peak_memory_reserved_gb,
        }])
        results_df = pd.concat([results_df, new_row], ignore_index=True)
        results_df.to_csv(csv_path, index=False)

        # === torch.save: raw per-run artifact, linked to the CSV row via run_id ===
        pt_path = os.path.join(OUT_DIR, f"{DATASET_NAME}_{dtype}_{file_timestamp}_{run_id}.pt")
        torch.save({
            "run_id": run_id,
            "dtype": dtype,
            "dataset": DATASET_NAME,
            "hellaswag_NLLs": hellaswag_NLLs,
            "accuracy_check": accuracy_check,
            "candidate_NLLs": all_candidate_NLLs,
        }, pt_path)

except Exception as e:
    traceback.print_exc()