import torch
import os
import glob
from model import GPT
import time
import traceback
import uuid
from datetime import datetime
import numpy as np
import argparse
import torch.nn as nn

DATA_DIR = "../../data"
OUT_DIR = "./output"
os.makedirs(OUT_DIR, exist_ok=True)

FINEWEB_10B = DATA_DIR + "/fineweb10B"
DATASET_NAME = "fineweb"
CALIBRATION_SHARD = "fineweb_train_000001.bin"  # [for saving] load_fineweb() also uses this value - pulled out as a constant so it's kept in the .pt file to later know which shard to reproduce/compare against; has no effect on the algorithm itself

def load_fineweb():
    path = FINEWEB_10B+"/"+CALIBRATION_SHARD
    data = np.memmap(path, dtype=np.uint16, mode='r', offset=256*4)
    return data

# -----------------------------------------------------------------------------
init_from = 'gpt2'

parser = argparse.ArgumentParser()
parser.add_argument('--dtype', type=str, required=True,
                     choices=['float64', 'float32', 'bfloat16', 'float16', 'float8_e4m3fn', 'float8_e5m2', 'int8'])
parser.add_argument('--quant_dtype', type=str, default='int8',
                     choices=['int8'])  # torch.iinfo() only supports integer dtypes, so int8 is currently the only valid option
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--resume', action='store_true')  # Find the most recently (mtime) saved file in OUT_DIR, restore layers that are already done, and continue from there
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
quant_dtype = {
    'int8': torch.int8
    }[args.quant_dtype]

BT = B*T
VAL_TOKENS = 10_485_760
fineweb_val = load_fineweb()[:VAL_TOKENS]


activation_inputs = {}

# Naive way
# def make_hook(layer_name):
#     def hook(module, input, output):
#         if layer_name not in activation_inputs:
#             activation_inputs[layer_name] = input[0].detach()
#         else:
#             activation_inputs[layer_name] = torch.cat([activation_inputs[layer_name], input[0].detach()], dim=0)
#     return hook

# Accumulation way - the same result as the naive way
def make_hook(layer_name):
    def hook(module, input, output):
        X = input[0].detach()
        X = X.view(-1,X.size(-1))
        H = 2*X.T@X
        if layer_name not in activation_inputs:
            activation_inputs[layer_name] = H
        else:
            activation_inputs[layer_name] += H
        raise StopIteration
    return hook

hook_handles = []

# === Quantization
# Hs_cholesky_R = {}
# Hs_cholesky_inverse = {}
quant_infos = {}

quant_dtype_info = torch.iinfo(quant_dtype)
damp_ratio = 0.01  # [for saving] the value itself (0.01) is the same as what was originally hardcoded; pulled into a variable so it's kept in the .pt file to later know which ratio was used
def quantization(weights, quant_dtype_info):
    # Scaler
    W_max, W_min = torch.max(weights, dim=-1, keepdim=True).values, torch.min(weights, dim=-1, keepdim=True).values
    Type_min, Type_max = quant_dtype_info.min, quant_dtype_info.max
    scaler = (W_max-W_min)/(Type_max-Type_min)
    zero_point = torch.clamp(torch.round(Type_min - W_min/scaler), Type_min, Type_max)
    # quantized_weight = torch.clamp(torch.round(weights/scaler) + zero_point, Type_min, Type_max)
    # dequantized_weight = scaler*(quantized_weight-zero_point)
    return scaler, zero_point # quantized_weight, scaler, zero_point

# def dequantization(quantized_weight, scaler, zero_point):
#     dequantized_weight = scaler.squeeze(-1)*(quantized_weight-zero_point.squeeze(-1))
#     return dequantized_weight


try:
    print("Dtype:", ptdtype)
    print("Quant Dtype:", quant_dtype)
    print("Batch Size:", B)
    torch.manual_seed(seed)

    model.eval()
    num_iter = len(fineweb_val)//(BT+1)
    linear_names = []
    run_id = uuid.uuid4().hex[:8]  # [for saving] id identifying this whole script run - shared between the per-handle files and the all_done file

    # ======= Resume - find the most recently (mtime) saved file of the same dtype in OUT_DIR, restore layers that are already done
    completed_layers = set()
    if args.resume:
        candidates = glob.glob(os.path.join(OUT_DIR, f"{DATASET_NAME}*Quantized_weights_{dtype}*.pt"))
        if candidates:
            resume_path = max(candidates, key=os.path.getmtime)  # based on actual mtime, not filename (filename string sort breaks once handle_idx reaches two digits)
            print("Resuming from:", resume_path)
            checkpoint = torch.load(resume_path)
            quant_infos = checkpoint["quant_infos"]
            for name, info in quant_infos.items():
                model.get_submodule(name).weight.data.copy_(info["weights"])
                completed_layers.add(name)
            print(f"Restored {len(completed_layers)} already-quantized layer(s), skipping them.")
        else:
            print("--resume given but no matching checkpoint found in", OUT_DIR, "- starting fresh.")
    # ======= Resume

    with torch.no_grad():
        for module_name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module_name not in completed_layers:
                handle = module.register_forward_hook(make_hook(module_name))
                hook_handles.append(handle)
                linear_names.append(module_name)
        global_handle_offset = len(completed_layers)  # [for logging] corrects for handle_idx restarting from 0 on --resume - not used at all in the GPTQ computation (hook_handles/linear_names indexing), only for log/save labeling
        for handle_idx in range(len(hook_handles)):
            global_handle_idx = handle_idx + global_handle_offset  # [for logging] actual sequence number across all layers - only used for print/save labeling below
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            CPU_time_start = time.perf_counter()
            total_tokens = 0  # [for logging] for per-token time calculation - total tokens actually fed into forward in this handle
            # === Activation acquisition
            for idx in range(num_iter+1):
                samples = fineweb_val[idx*(BT+1):(idx+1)*(BT+1)]
                _B = (len(samples)-1)//T
                if _B<=0:
                    break
                samples = torch.tensor(samples[:(_B*T+1)].astype(np.int64))
                x, y = samples[:-1].view(_B, -1).to(device), samples[1:].view(_B, -1).to(device)
                total_tokens += _B*T  # [for logging]

                try:
                    logits = model(x, y) # [batch_size, Token_size, vocab_size] - probabilities
                except StopIteration:
                    pass

            torch.cuda.synchronize()  # [for logging] confirms the forward section has actually finished - without this, since the GPU runs asynchronously, forward compute could bleed into the timing of the next section
            forward_time_end = time.perf_counter()
            forward_time = forward_time_end - CPU_time_start  # [for logging] time spent only on model forward (activation acquisition)
            quant_time_start = time.perf_counter()  # [for logging]

            # === Quantization calculation
            module_name = linear_names[handle_idx]
            target = model.get_submodule(module_name)
            W = target.weight
            W_original = W.detach().clone()  # [for saving] not used in the quantize algorithm - needed later to compare against the original and review error
            H = activation_inputs[module_name]
            r_size, _ = H.size()
            H[range(r_size), range(r_size)] += torch.diag(H).mean()*damp_ratio # Dampening
            L = torch.linalg.cholesky(H)
            inv_H = torch.cholesky_inverse(L)
            inv_R = torch.linalg.cholesky(inv_H, upper=True)
            # Deriven delta_F form
            # quantized_W, scaler, zero_point = quantization(W, quant_dtype_info)
            scaler, zero_point = quantization(W, quant_dtype_info)
            Type_min, Type_max = quant_dtype_info.min, quant_dtype_info.max
            Q_int = torch.zeros_like(W, dtype=quant_dtype)  # [for saving] not used in the quantize algorithm - filled column by column with the actual int8 integer codes, needed later for real compressed storage
            layer_error = 0.0  # [for logging] not used in the quantize algorithm - accumulated value for reviewing how much this layer drifted from quantization
            for quant_idx in range(W.size(-1)):
                # Deriven delta_F form
                # First_R = inv_R[quant_idx,quant_idx:]
                w = W[:, quant_idx]
                q = torch.clamp(torch.round(w/scaler.squeeze(-1)) + zero_point.squeeze(-1), Type_min, Type_max)
                dq = scaler.squeeze(-1)*(q - zero_point.squeeze(-1))
                Q_int[:, quant_idx] = q  # [for saving]
                layer_error += ((w - dq)**2).sum().item()  # [for logging]
                err = (w - dq) / inv_R[quant_idx, quant_idx]
                W[:, quant_idx:] -= torch.outer(err, inv_R[quant_idx, quant_idx:])
                # Deriven delta_F form
                # delta_F = torch.outer((W[:, quant_idx] - quantized_W[:, quant_idx])/First_R[0]**2, First_R[0]*First_R)
                # W[:, quant_idx:] -= delta_F
            quant_infos[module_name] = dict(
                handle_idx=global_handle_idx,  # [for logging] actual sequence number across all layers (with --resume offset applied)
                weights=W,                    # final quantize+dequantize weight (float)
                weights_original=W_original,  # [for saving] original weight before quantize
                quantized_int=Q_int,          # [for saving] actual int8 integer codes
                scaler=scaler,
                zero_point=zero_point,
                layer_error=layer_error,      # [for logging] sum of this layer's quantize error (sum of per-column (w-dq)^2)
            )

            torch.cuda.synchronize()
            quant_time_end = time.perf_counter()
            quant_time = quant_time_end - quant_time_start  # [for logging] time spent only on the GPTQ computation (Cholesky + quantize column loop)
            CPU_time_end = quant_time_end
            elapsed_time = CPU_time_end - CPU_time_start  # original total time measurement - should be nearly identical to forward_time+quant_time
            time_per_token = elapsed_time / total_tokens  # [for logging] for comparing actual throughput when batch_size changes
            remaining_handles = len(hook_handles) - (handle_idx+1)
            eta = elapsed_time * remaining_handles  # [for logging] estimated time remaining for the remaining handles, based on this handle
            print(f"{global_handle_idx}th handle 100% Done - Total elapsed: {elapsed_time:.1f}s "
                  f"(Forward: {forward_time:.1f}s, Quant: {quant_time:.1f}s), "
                  f"Tokens: {total_tokens}, Time/token: {time_per_token*1000:.4f} ms, ETA: {eta:.1f}s")
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M")
            file_timestamp = now.strftime("%Y%m%d-%H%M%S")  # down to the second - used in the filename so handles don't collide

            # === torch.save: raw per-run artifact. Filename is time-based instead of run_id - --resume finds the latest file by mtime ===
            pt_path = os.path.join(OUT_DIR, f"{DATASET_NAME}_{global_handle_idx}th_Quantized_weights_{dtype}_{file_timestamp}.pt")
            torch.save({
                "file_timestamp": file_timestamp,
                "run_id": run_id,
                "dtype": dtype,
                "quant_dtype": quant_dtype,
                "batch_size": B,  # [for saving]
                "dataset": DATASET_NAME,
                "calibration_shard": CALIBRATION_SHARD,  # [for saving]
                "damp_ratio": damp_ratio,  # [for saving]
                "elapsed_time_sec": elapsed_time,
                "forward_time_sec": forward_time,  # [for logging]
                "quant_time_sec": quant_time,  # [for logging]
                "total_tokens": total_tokens,  # [for logging]
                "time_per_token_sec": time_per_token,  # [for logging]
                "quant_infos": quant_infos,
            }, pt_path)

            hook_handles[handle_idx].remove()

except Exception as e:
    traceback.print_exc()

# === torch.save: raw per-run artifact. Filename is time-based instead of run_id - --resume finds the latest file by mtime ===
pt_path = os.path.join(OUT_DIR, f"{DATASET_NAME}_Quantized_weights_{dtype}_{file_timestamp}_all_done.pt")
torch.save({
    "file_timestamp":file_timestamp,
    "run_id": run_id,
    "dtype": dtype,
    "quant_dtype": quant_dtype,
    "batch_size": B,  # [for saving]
    "dataset": DATASET_NAME,
    "calibration_shard": CALIBRATION_SHARD,  # [for saving]
    "damp_ratio": damp_ratio,  # [for saving]
    "elapsed_time_sec": elapsed_time,
    "forward_time_sec": forward_time,  # [for logging]
    "quant_time_sec": quant_time,  # [for logging]
    "total_tokens": total_tokens,  # [for logging]
    "time_per_token_sec": time_per_token,  # [for logging]
    "quant_infos": quant_infos,
}, pt_path)
