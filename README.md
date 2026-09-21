# AI-Experiments

Experiment logs, scripts, and results from a personal study of **AI efficiency** —
specifically, how to *measure* a language model's resource usage (GPU memory so far,
with compute/time efficiency planned next).

This repository is a companion to my blog post series on the topic, where the
experiments and findings recorded here are written up in more detail.

## Based on nanoGPT

The code in this repository is based on [nanoGPT](https://github.com/karpathy/nanoGPT)
by [Andrej Karpathy](https://github.com/karpathy). The model and training code
(`model.py`, `train.py`, and scripts derived from them) originate from that project;
any modifications here exist only to instrument/measure resource usage for this study,
not to change the model itself.

nanoGPT is released under the MIT License. This repository reuses/derives from that
code under the same license — see [LICENSE](./LICENSE), which retains Andrej Karpathy's
original copyright notice as required by the MIT License terms. All credit for the
original nanoGPT implementation belongs to him.

## Contents

- [`Kernel investigation/`](./Kernel%20investigation/README.md) — investigates
  whether CUDA kernel selection for nanoGPT's GPT-2 forward pass is sensitive to
  tensor dimension alignment (vocab size, batch size, sequence padding), by
  capturing `torch.profiler` traces across combinations and comparing which
  kernels get dispatched. Run on Google Colab; see its README for setup and
  details.
- [`KV Cache/`](./KV%20Cache/README.md) — KV cache implementation for nanoGPT's
  GPT-2, correctness verification, and latency measurements (end-to-end → per-token →
  per-self-attention-layer → per-kernel via Nsight Compute). Run on Google Colab; see
  its README for setup and experiment order.
- [`Memory measurement/`](./Memory%20measurement/README.md) — GPU memory profiling
  of nanoGPT's GPT-2 on its first validation batch, using
  `torch.cuda.memory._record_memory_history` / `_dump_snapshot` to capture allocator
  snapshots (viewable at [pytorch.org/memory_viz](https://pytorch.org/memory_viz)).
  `First_batch/` snapshots the whole model's forward + loss; `One_selfattention/`
  isolates just block 0's self-attention layer via forward hooks. Run on Google
  Colab; see its README for setup and details.
- [`Quantization/`](./Quantization/README.md) — investigates how much
  post-training quantization degrades nanoGPT's GPT-2, starting by establishing
  how far dtype precision alone can be pushed before HellaSwag/FineWeb accuracy
  and perplexity visibly drop, then applying real quantization from that point.
  **Work in progress.** Run on Google Colab; see its README for setup and
  details.
- [`Batch Size vs Inference Speed/`](./Batch%20Size%20vs%20Inference%20Speed/README.md) —
  investigates why `batch_size` didn't affect GPT-2's pure forward inference
  speed, motivated by an unexpected finding during the quantization work above
  (`batch_size=1` vs `64` showed no meaningful timing difference). **Root cause
  confirmed: GPU power-cap clock throttling** (T4's 70W limit), not kernel/compute
  inefficiency — verified with a controlled `nvidia-smi -lgc` clock-lock
  experiment at three scopes (whole model, single block, single kernel) and
  cross-checked against `ncu`. Run on Google Colab; see its README for setup,
  findings, and details.

## License

MIT, inherited from nanoGPT. See [LICENSE](./LICENSE).
