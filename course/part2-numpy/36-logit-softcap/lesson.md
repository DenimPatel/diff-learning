QK-norm (lesson 29) bounds attention logits by normalizing queries and keys. **Soft-capping** bounds the logits themselves, smoothly:

```
softcap(x) = cap · tanh(x / cap)
```

For `|x| ≪ cap` this is almost exactly `x`, and for large `|x|` it flattens out at `±cap`. Unlike a hard clip, the gradient never becomes exactly zero.

### The diff
- `Tensor.tanh()` is a new operation: `d tanh(x) / dx = 1 − tanh(x)²`.
- Attention logits are capped **before** the causal mask is added, so masked positions still get `−1e9`.
- The output logits are capped before the loss. `max_logit` charts the largest one.

### Why
Without a cap, nothing stops logits from growing. Large output logits make the model overconfident, and large attention logits saturate softmax (zero gradient, loss spikes). Gemma 2 caps attention logits at 50 and final logits at 30, and Grok-1 caps attention at 30. Those are far above typical values, so the cap acts only as a safety rail.

### What to try
The defaults (20 and 15) are well above anything this model produces, so the loss matches lesson 21. The interesting experiments are the tight caps. **Output cap too tight** limits how confident a prediction can be, which puts a floor under the loss. **Attention cap too tight** flattens attention. A cap is a constraint on what the model can express, not a free lunch.

**Up to date:** Gemma 2 used soft-capping, and Gemma 3 replaced the attention cap with QK-norm, which bounds the same quantity more cheaply and works with fast attention kernels (a tanh inside FlashAttention is awkward). Output-logit capping and z-loss (lesson 46) remain common for keeping the output layer in check.
