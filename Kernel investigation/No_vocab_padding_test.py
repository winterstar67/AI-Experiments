import torch
import math
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import torch.nn.functional as F
import time
from torch.profiler import profile, ProfilerActivity
import json
import traceback
import gc

DATA_DIR = "../data"
OUT_DIR = "../output"
os.makedirs(OUT_DIR, exist_ok=True)

HELLASWAG_DIR = DATA_DIR + "/hellaswag (Quantization)"

def load_hellaswag():
    path = HELLASWAG_DIR+"/tokenized_hellaswag_val.json"
    with open(path, "r") as f:
        examples = json.load(f)
    return examples

def get_lens(token_lists:list):
    return list(map(lambda x: len(x), token_lists))


# -----------------------------------------------------------------------------
init_from = 'gpt2' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')

config = {
    "batch_size": "26 and 25 both",
    "experiment_repeat_times": 1,
    "seed": 1337,
    "device": "cuda",
    "dtype": 'float16 and float32 both', # Change this to check kernel by dtype 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16',
    "compile": False,
    "torch.backends.cuda.matmul.allow_tf32": True, # allow tf32 on matmul
    "torch.backends.cudnn.allow_tf32": True, # allow tf32 on cudnn
    "torch.use_deterministic_algorithms": False
    }

experiment_repeat_times = config['experiment_repeat_times']
seed = config['seed']
device = config['device'] # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
cuda_allow_tf32 = config['torch.backends.cuda.matmul.allow_tf32']
cudnn_allow_tf32 = config['torch.backends.cudnn.allow_tf32']
deterministic_kernel = config['torch.use_deterministic_algorithms']

compile = config['compile'] # use PyTorch 2.0 to compile the model to be faster
# exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = cuda_allow_tf32 # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = cudnn_allow_tf32 # allow tf32 on cudnn
torch.use_deterministic_algorithms(deterministic_kernel)
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast

dtype = config['dtype'] # 'float32' or 'bfloat16' or 'float16'

model = GPT.from_pretrained(init_from, dict(dropout=0.0))
model.to(device)

try:
    for dtype in ["float32", "float16"]:
        ptdtype = {'float64': torch.float64, 'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

        for batch_size, padding_type in [[25, 'odd'],[26, 'even']]:
            print("Dtype:", dtype)
            print("Batch size:", batch_size)
            print("Padding:", padding_type)
            torch.manual_seed(seed)

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

            if compile:
                model = torch.compile(model) # requires PyTorch 2.0 (optional)

            model.eval()
            num_iter = len(hella_swag_val)//batch_size
            accuracy_check, hellaswag_NLLs = [], []

            with torch.no_grad(), ctx:
                torch.cuda.synchronize()
                CPU_time_start = time.perf_counter()
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    with_modules=True,
                    with_stack=True,
                    record_shapes=True
                ) as prof:
                    for idx in range(num_iter+1):
                        if (idx==num_iter) and (len(hella_swag_val)%batch_size==0):
                            break
                        samples = hella_swag_val[idx*batch_size:(idx+1)*batch_size]
                        samples_len = len(samples)
                        candidates = [i for x in samples for i in x['candidates']]
                        masking_info = [i for x in samples for i in x['masking_info']]
                        tokens_len = [len(x) for x in candidates]
                        max_token_len = max(tokens_len)
                        if padding_type=="even":
                            padding_size=16
                            max_token_len += ((padding_size-max_token_len%padding_size)%padding_size+1)
                        else:
                            if max_token_len%2==1:
                                max_token_len += 1                                     
                        x = torch.zeros((samples_len*4, max_token_len),dtype=torch.int64)
                        padded_masking = torch.zeros((samples_len*4, max_token_len-1),dtype=torch.int32)
                        for _idx in range(samples_len*4):
                            x[_idx, :tokens_len[_idx]] = torch.tensor(candidates[_idx])
                            padded_masking[_idx, :tokens_len[_idx]-1] = torch.tensor(masking_info[_idx])[1:]
                        padded_masking = padded_masking.to(device)
                        x = x.to(device)
                        y = x[:, 1:].contiguous()
                        x = x[:, :-1].contiguous()
                        logits, _ = model(x, y) # [batch_size, Token_size, vocab_size] - probabilities
                        probabilities = torch.softmax(logits, dim=-1)
                        label_probabilities = torch.gather(probabilities, dim=-1, index=y.unsqueeze(-1)).squeeze(-1)
                        NLL = -1*torch.sum(torch.log(label_probabilities)*padded_masking, dim=-1)/padded_masking.sum(dim=-1)

                        # Grouping
                        predicted_label_list = []
                        for _idx in range(samples_len):
                            sample_NLLs = NLL[_idx*4:(_idx+1)*4]
                            predicted_label = torch.argmin(sample_NLLs)
                            sample_NLL = sample_NLLs[samples[_idx]['label']]
                            predicted_label_list.append(predicted_label)
                            if predicted_label==samples[_idx]['label']:
                                accuracy_check.append(1)
                            else:
                                accuracy_check.append(0)
                            hellaswag_NLLs.append(sample_NLL.item())
                        if idx == 3:
                            print(f"accuracy on {len(accuracy_check)} samples:", sum(accuracy_check)/len(accuracy_check))
                            print("Input token length is", x.shape)
                            print("target token length is", y.shape)
                            break
                torch.cuda.synchronize()
                prof.export_chrome_trace(f"{OUT_DIR}/No_vocab_padding__Dtype_{dtype}__BatchSize_{batch_size}__input_shape_{list(x.shape)}_trace_result.json")
                del prof, x,y, logits, padded_masking, probabilities, label_probabilities, NLL
                gc.collect()
                torch.cuda.empty_cache()
                print("\n\n")

except Exception as e:
    traceback.print_exc()