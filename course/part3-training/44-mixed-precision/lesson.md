GPUs are many times faster at 16-bit math than at 32-bit, and 16-bit tensors take half the memory and bandwidth. Training in **mixed precision** means computing in 16 bits wherever it is safe and keeping 32 bits where it isn't. NumPy has no bf16, so this lesson **simulates** it: `round_to()` rounds a float64 array to the nearest value the chosen format can represent.

### The formats
| format | exponent bits | mantissa bits | range | relative precision |
|---|---|---|---|---|
| fp32 | 8 | 23 | 1e±38 | ~1e-7 |
| **bf16** | 8 | 7 | 1e±38 | ~0.4% |
| fp16 | 5 | 10 | 6e-8 to 65504 | ~0.05% |

bf16 is float32 with the low 16 bits cut off: same range, much less precision. fp16 is more precise, but values outside its small range overflow to `inf` or underflow to `0`.

### The diff
- Every `Tensor` the forward pass creates is stored rounded (`Tensor.__init__`), and every gradient the backward pass produces is rounded too.
- **Master weights**: the optimizer updates a full-precision copy. Why? A bf16 weight of 0.08 can only change in steps of ~0.0003. An update smaller than half of that rounds away to *nothing*. Set `master_weights = false` and the weights are re-rounded after each update.
- **Loss scaling** (needed for fp16): small gradients underflow to 0 in fp16. Multiplying the loss by `loss_scale` before backward scales every gradient up into fp16's range, and dividing afterwards undoes it. `backward(scale)` starts the backward pass at `scale` instead of 1.

### What to try
Compare **No master weights, small LR** with **Master weights, small LR**: same format, but the first silently drops updates. The default bf16 run matches **Full precision** within noise. That robustness is why bf16 replaced fp16 for LLM training: its fp32-sized range makes loss scaling unnecessary.

**Up to date:** LLMs are trained in bf16 with fp32 master weights and fp32 optimizer state. Frontier runs are moving matmuls to **FP8** (DeepSeek-V3 with fine-grained scaling, many 2025–26 models), and to MXFP4/NVFP4 on the newest GPUs. Each step down needs new tricks for the same two problems seen here: range and lost updates.
