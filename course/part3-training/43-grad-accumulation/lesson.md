Bigger batches give less noisy gradients, and large training runs use batches of millions of tokens. But activation memory grows with the batch: every intermediate value of the forward pass is kept for the backward pass. **Gradient accumulation** decouples the two. It splits the batch into `accum_steps` micro-batches, runs forward and backward on each in turn, **adds up** the gradients, and takes a single optimizer step at the end.

### The diff
- `micro_batch = batch_size // accum_steps` sequences are in memory at once.
- Each micro-batch loss is divided by `accum_steps`, so the summed gradient is the gradient of the **mean** loss over the whole batch.
- `backward()` already adds into `p.grad` (`child.grad = child.grad + …`), so accumulating needs nothing new. We just don't zero the gradients between micro-batches, only after the optimizer step. In PyTorch this is the reason for `optimizer.zero_grad()`.
- Before training, the script checks that 4 micro-batches give **the same gradient** as one full batch: the difference is ~1e-17, which is floating-point rounding.

### The fine print
It is exactly equivalent here because the loss is a mean over independent sequences. **BatchNorm**, which mixes statistics across a batch, would break the equivalence (transformers use LayerNorm/RMSNorm, so they're safe). With **padding**, dividing by `accum_steps` instead of by the true token count weights micro-batches unequally, a real bug that was fixed in Hugging Face Transformers in 2024.

### What to try
**No accumulation** gives the same curve. **Batch 128 from micro-batches of 8** gives a 4× bigger batch for the memory of 8 sequences, at 4× the compute per step.

**Up to date:** every large run accumulates gradients. Combined with data parallelism, the global batch is `micro_batch × accum_steps × number of GPUs`. Runs also often *grow* the batch during training (batch-size warmup), because a bigger batch helps more once the loss is lower.
