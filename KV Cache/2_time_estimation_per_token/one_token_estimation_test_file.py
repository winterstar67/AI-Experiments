import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import torch.nn.functional as F
import time

# -----------------------------------------------------------------------------
init_from = 'gpt2' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'

config = {
    "batch_size": 16,
    "start_token_size": 1014,
    "warmup_steps": 10,
    "experiment_repeat_times": 10,
    "max_new_tokens": 1024,
    "temperature": 0.8,
    "top_k": 10,
    "seed": 1337,
    "device": "cuda",
    "dtype": 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16',
    "compile": False,
    "torch.backends.cuda.matmul.allow_tf32": True, # allow tf32 on matmul
    "torch.backends.cudnn.allow_tf32": True, # allow tf32 on cudnn
    "torch.use_deterministic_algorithms": False
    }


start_token_size = config['start_token_size']
max_new_tokens = config['max_new_tokens'] # number of tokens generated in each sample
temperature = config['temperature'] # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = config['top_k'] # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = config['seed']
device = config['device'] # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = config['dtype'] # 'float32' or 'bfloat16' or 'float16'
cuda_allow_tf32 = config['torch.backends.cuda.matmul.allow_tf32']
cudnn_allow_tf32 = config['torch.backends.cudnn.allow_tf32']
deterministic_kernel = config['torch.use_deterministic_algorithms']

compile = config['compile'] # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = cuda_allow_tf32 # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = cudnn_allow_tf32 # allow tf32 on cudnn
torch.use_deterministic_algorithms(deterministic_kernel)
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float64': torch.float64, 'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


# init from a given GPT-2 model
model = GPT.from_pretrained(init_from, dict(dropout=0.0))
if dtype == "float64":
    model = model.double()

if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# ok let's assume gpt-2 encodings by default
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

KV_SAVE_DIR = "Time_estimation_result"
os.makedirs(KV_SAVE_DIR, exist_ok=True)

# run generation
model.eval()
model.to(device)
warmup_steps = config['warmup_steps']
experiment_repeat_times = config['experiment_repeat_times']

with open("../data/Shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

batch_size = config['batch_size']
start_tokens = []
encoded_text = encode(text)
for i in range(batch_size):
    start_tokens.append(encoded_text[i*start_token_size:(i+1)*start_token_size])
start_data = (torch.tensor(start_tokens, dtype=torch.long, device=device)).view(batch_size,-1)

counter = {"inf_no_cache":0, "inf_cache":0}

time_results = {"inf_no_cache":{}, "inf_cache":{}}
for _key in time_results.keys():
    for _step in range(start_token_size, max_new_tokens):
        time_results[_key][_step] = []

with torch.no_grad():
    for running_mode in ["inf_no_cache", "inf_cache"]:
        for repeat_time in range(experiment_repeat_times):
            print(f"===== running_mode = {running_mode} =====") 
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            x = start_data

            with ctx:
                for _ in range(warmup_steps):
                    _ = model(x, running_mode)
                for layer_idx, block in enumerate(model.transformer.h):
                    if running_mode == "inf_cache":
                        block.attn.reset_KV()

                for _step in range(start_token_size, max_new_tokens):
                    start_evt = torch.cuda.Event(enable_timing=True)
                    end_evt   = torch.cuda.Event(enable_timing=True)

                    idx_cond = x if x.size(1) <= model.config.block_size else x[:, -model.config.block_size:]
                    exist_KV_size = x.size(1)
                    start_evt.record()
                    logits, _ = model(idx_cond, running_mode=running_mode)
                    end_evt.record()
                    time_results[running_mode][_step].append((start_evt,end_evt))

                    logits = logits[:, -1, :] / temperature
                    max_index = torch.argmax(logits, dim=1)

                    if top_k is not None:
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = -float('Inf')

                    probs = F.softmax(logits, dim=-1)

                    idx_next = torch.argmax(probs, dim=-1, keepdim=True)

                    if (running_mode == 'inf_no_cache' and counter['inf_no_cache'] <=0) or (running_mode == 'inf_cache' and counter['inf_cache'] <=0):
                        decoded_next = [decode([idx_next[b, 0].item()]) for b in range(idx_next.size(0))]
                        print(f"[{running_mode}] step {_step}: next tokens = {decoded_next}")
                    x = torch.cat((x, idx_next), dim=1)

                torch.cuda.synchronize()
                if running_mode =='inf_no_cache':
                    counter['inf_no_cache'] += 1
                else:
                    counter['inf_cache'] += 1

            for layer_idx, block in enumerate(model.transformer.h):
                if running_mode == "inf_cache":
                    block.attn.reset_KV()

for _key in time_results.keys():
    print("============== running mode:", _key)
    for _step in range(start_token_size, max_new_tokens):
        for repeat in range(experiment_repeat_times):
           time_results[_key][_step][repeat] = time_results[_key][_step][repeat][0].elapsed_time(time_results[_key][_step][repeat][1])
        print(_step,"Step time:", time_results[_key][_step])

torch.save(
        {
            "time_result": time_results,
            "config": config
        },
        os.path.join(KV_SAVE_DIR, f"one_token_estimation_result.pt")
    )