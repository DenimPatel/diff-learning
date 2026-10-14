Full fine-tuning updates every weight. For a 70B-parameter model that means gradients and Adam state for 70B numbers (several times the model's memory), and a full 140 GB copy of the model for every task. **LoRA** (Low-Rank Adaptation) observes that the *change* a fine-tune makes to a weight matrix has low rank, so it learns the change directly in factored form:

```
W' = W + (alpha / rank) · A @ B        A: (n_in, rank)   B: (rank, n_out)
```

`W` is frozen. Only `A` and `B` are trained. For a 4096×4096 matrix with `rank = 16` that is 131k numbers instead of 16.8M, **0.8%**.

### The diff
- `linear()` now checks the `lora` dict. If the weight has an adapter, it adds `(x @ A) @ B * scale`. It never forms the full `A @ B` matrix.
- After pretraining (and after adding `=`), every matrix selected by `lora_targets` gets an adapter. `A` is random and **`B` starts at zero**, so at step 0 the model is exactly the base model and fine-tuning starts from where pretraining ended.
- `params` becomes the adapters only, so the optimizer state shrinks with them. Our toy autograd still computes gradients for the frozen weights and discards them. PyTorch skips them entirely (`requires_grad=False`), which is where much of the memory saving comes from.
- `lora_scale = alpha / rank` lets you change the rank without re-tuning the learning rate. LoRA typically wants a **higher learning rate** than full fine-tuning, because `B` has to grow from zero.

### After training
- **Detach** the adapters (`lora.clear()`): the base model is back, untouched. One base model can serve many tasks, each with its own few-KB adapter swapped in per request.
- **Merge** them: `W += scale · A @ B` gives the same outputs (max change ~1e-14) with no extra cost at inference time.

### What to try
Our model is tiny, so even rank 4 is 18% of its size, and the comparison with full SFT (lesson 51) is not at real scale. Still, the trends are real: **Rank 1** is too small, and **Attention only** shows that the MLPs matter. Notice that `names` still rises while the adapter is on: LoRA forgets *less* than full fine-tuning at scale, not *nothing*. Its real superpower is that the base weights are never touched.

**Up to date:** LoRA and QLoRA (lesson 54) are the default way to fine-tune open models on a single GPU. Recent work (*LoRA Without Regret*) finds LoRA matches full fine-tuning for SFT and RL when it is applied to **all** layers, MLPs included, and given a ~10× higher learning rate. That is what we see here. Variants include DoRA (separate magnitude and direction) and rsLoRA (scale by α/√r).
