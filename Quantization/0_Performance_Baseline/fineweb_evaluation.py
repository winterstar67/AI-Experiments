import torch
import os
from model import GPT
import time
import traceback
import pandas as pd
import uuid
from datetime import datetime
import numpy as np
import argparse

DATA_DIR = "../data"
OUT_DIR = "./output"
os.makedirs(OUT_DIR, exist_ok=True)

FINEWEB_10B = DATA_DIR + "/fineweb10B"
DATASET_NAME = "fineweb"

def load_fineweb():
    path = FINEWEB_10B+"/fineweb_val_000000.bin"
    data = np.memmap(path, dtype=np.uint16, mode='r', offset=256*4)
    return data

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
    "token_size": 1024,
    "warmup": 10,
    "seed": 1337,
    "device": "cuda",
    "dtype": args.dtype,
    "torch.backends.cuda.matmul.allow_tf32": False,
    "torch.backends.cudnn.allow_tf32": False,
    "torch.use_deterministic_algorithms": True
    }

seed = config['seed']
B = config['batch_size']
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

model = GPT.from_pretrained(init_from, dict(dropout=0.0, use_flash=args.use_flash))
model.to(device=device, dtype=ptdtype)


BT = B*T
VAL_TOKENS = 10_485_760
fineweb_val = load_fineweb()[:VAL_TOKENS]

try:
    print("Dtype:", ptdtype)
    torch.manual_seed(seed)

    model.eval()
    num_iter = len(fineweb_val)//(BT+1)
    print_interval = max(num_iter//(10),1)

    fineweb_NLLs = []

    # ======= Warmup sample preparation
    samples = torch.tensor(fineweb_val[:(BT+1)].astype(np.int64))
    x, y = samples[:-1].view(B, -1).to(device), samples[1:].view(B, -1).to(device)
    # ======= Warmup sample preparation


    with torch.no_grad():
        # ======= Warmup stage
        for _ in range(warmup):
            logits = model(x,y)
        # ======= Warmup stage
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        CPU_time_start = time.perf_counter()
        for idx in range(num_iter+1):
            samples = fineweb_val[idx*(BT+1):(idx+1)*(BT+1)]
            _B = (len(samples)-1)//T
            if _B<=0:
                break
            samples = torch.tensor(samples[:(_B*T+1)].astype(np.int64))
            x, y = samples[:-1].view(_B, -1).to(device), samples[1:].view(_B, -1).to(device)
            masking = (y != 50256).to(device)

            logits = model(x, y) # [batch_size, Token_size, vocab_size] - probabilities
            logits_f = logits.float() # low-precision dtypes can cause log(0.0)=-inf errors, so upcast to float for precision
            m = torch.max(logits_f, dim=-1, keepdim=True).values # max value to prevent exp() overflow
            log_Z = m.squeeze(-1) + torch.log(torch.sum(torch.exp(logits_f - m), dim=-1)) # log-sum-exp
            label_logits = torch.gather(logits_f, dim=-1, index=y.unsqueeze(-1)).squeeze(-1)
            label_log_probabilities = label_logits - log_Z # log(softmax(x)) = x - logsumexp(x)
            NLL = -1*torch.sum(label_log_probabilities*masking, dim=-1)/masking.sum(dim=-1)

            fineweb_NLLs.extend(NLL.tolist())
            if idx%print_interval==0:
                print(f"{round(100*idx/num_iter, 1)}% Done - Average NLL:", sum(fineweb_NLLs)/len(fineweb_NLLs))
        torch.cuda.synchronize()
        print(f"100% Done - Average NLL:", sum(fineweb_NLLs)/len(fineweb_NLLs))
        CPU_time_end = time.perf_counter()
        peak_memory_allocated_gb = torch.cuda.max_memory_allocated() / (1024**3)
        peak_memory_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)

        elapsed_time = CPU_time_end - CPU_time_start
        avg_NLL = sum(fineweb_NLLs) / len(fineweb_NLLs)
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
            "token_size": T,
            "batch_size": B,
            "num_samples": len(fineweb_NLLs),
            "elapsed_time_sec": elapsed_time,
            "avg_NLL": avg_NLL,
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
            "fineweb_NLLs": fineweb_NLLs,
        }, pt_path)

except Exception as e:
    traceback.print_exc()