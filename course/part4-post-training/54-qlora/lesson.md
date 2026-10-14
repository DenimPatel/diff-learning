LoRA shrinks the *trainable* part, but the frozen base still has to sit in memory: 140 GB for a 70B model in 16-bit. **QLoRA** notices that the base is only ever *read*. It is never updated, so it can be stored in **4 bits**, while the small adapters keep full precision. A 65B model then fine-tunes on a single 48 GB GPU.

### The diff
Before training, every matrix that carries an adapter is quantized with `quantize_4bit` (absmax per group of `quant_group` weights, as in lesson 53). The forward pass computes `x @ dequant(W_4bit) + (x @ A) @ B · scale`, and gradients flow *through* the quantized weights into `A` and `B`. Nothing else changes from lesson 52.

### NF4
The QLoRA paper's data type, **4-bit NormalFloat**, puts its 16 levels at the quantiles of a normal distribution instead of spacing them evenly. Trained weights are roughly normal, so each level is used about equally often. That is the information-theoretically efficient choice, and it shows up as a slightly smaller loss increase than `int4`. QLoRA also uses **double quantization** (the scales themselves are quantized to 8 bits) and **paged optimizers** to survive memory spikes. Both are memory tricks that don't change the math.

### What to try
Compare **No quantization** with the default: same exact match, a quarter of the memory for the frozen weights. Then compare `names val loss of the base` for `nf4` and `int4`.

**Up to date:** QLoRA is how most people fine-tune open-weight models on consumer GPUs (bitsandbytes, Unsloth, Axolotl, TRL). A merged QLoRA model is usually dequantized to 16-bit, merged, and then re-quantized for serving.
