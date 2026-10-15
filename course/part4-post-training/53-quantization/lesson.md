Generating each token means reading **every weight** from memory, so inference speed is bound by memory bandwidth, not arithmetic. Fewer bytes per weight means a faster model that fits on smaller hardware. **Quantization** stores each weight as a small integer `q` and keeps one float `scale` for a block of weights:

```
w ≈ scale · q        scale = max|w| / (2^(bits-1) − 1)        q = round(w / scale)
```

This is *absmax*, symmetric quantization. Rounding is the only error, and it is at most `scale / 2` per weight.

### Granularity
The question is how many weights share one scale, set by `granularity`:
- **tensor**: one scale per matrix. The cheapest option, but one large weight inflates the scale for all the others, and the small ones round to 0.
- **channel**: one scale per output column.
- **group**: one scale per `group_size` consecutive inputs of a column, as used by GPTQ/AWQ/GGUF formats (groups of 32–128). Scales take extra memory, which the size column includes as 16 bits each.

`set_precision()` quantizes the transformer blocks and then dequantizes them for the forward pass, which is what "fake quantization" in a simulator does. Real kernels keep the int4 values packed and dequantize them on the fly in fast on-chip memory. Embeddings and `lm_head` stay in high precision, as most recipes leave them.

### Outliers
LLMs develop a few **outlier channels** with values 10–100× larger than the rest, and those broke naive int8 (LLM.int8()). `outlier_scale` plants one without changing what the model computes: `relu(s·z) = s·relu(z)`, so multiplying a column of `mlp_fc1` by `s` and dividing the matching row of `mlp_fc2` by `s` gives the same function. Run **An outlier channel, per-tensor**: same function, but now int4 hurts. Then run the **group-wise** version to see the damage contained. SmoothQuant and AWQ use the same identity in reverse to *remove* outliers before quantizing.

### What to try
Read the table from top to bottom. int8 is free, int4 group-wise costs almost nothing, and int2 is where naive methods fall apart. Set `bits = 2` and read the samples.

**Up to date:** open models are routinely served at 8 or 4 bits (GGUF Q4_K_M, GPTQ, AWQ, FP8 on newer GPUs), and some are now **trained** for low precision (quantization-aware training, MXFP4 in gpt-oss). Better rounding (GPTQ uses second-order information to compensate for each rounding error) makes 3-bit usable on large models.
