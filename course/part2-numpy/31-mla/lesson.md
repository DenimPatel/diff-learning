GQA (lesson 23) shrinks the KV cache by sharing key/value heads, which costs some quality. **Multi-head latent attention** (DeepSeek-V2/V3, Kimi K2, GLM-5, Mistral Large 3) takes another route: **low-rank compression**. Each token is projected down to a small latent vector `c_kv`, and keys and values for *every* head are projected up from it:

```
c_kv = x @ W_down              (B, T, kv_latent)     <- the only thing the cache stores
k    = c_kv @ W_k_up           (B, T, n_embd) -> heads
v    = c_kv @ W_v_up
```

### The diff
`attn_wk` and `attn_wv` become one shared down-projection plus two up-projections. Queries and the rest of the block are unchanged. The up-projections get `std = 1/√kv_latent` so that keys and values start at a similar scale as before. The printout compares the per-token cache: `n_layer × kv_latent` numbers instead of `2 × n_layer × n_embd`.

### Why it's cheap at inference
It looks as if we must re-expand every cached latent into keys at every step. We don't, because matrix products associate: `q · k = q · (c W_k_up) = (q W_k_upᵀ) · c`. The up-projection can be **absorbed** into the query projection, and `W_v_up` into the output projection, so attention runs directly against the latents. DeepSeek also keeps a small separate key part for RoPE ("decoupled RoPE"), because the rotation does not commute with the absorption trick. Our model has no RoPE, so we skip it.

### What to try
Compare **Latent of 2** and **Latent of 32** with the default. Even heavy compression costs little here, because each head's keys and values are highly redundant with the other heads'. MLA bets on exactly this.

**Up to date:** DeepSeek reports MLA matching or beating standard multi-head attention with a 93% smaller KV cache. It is now used well beyond DeepSeek, and methods like TransMLA convert existing GQA models to MLA after training.
