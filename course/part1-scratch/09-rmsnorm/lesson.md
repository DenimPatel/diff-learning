Every residual block *adds* to the stream `x`, so its magnitude drifts as layers stack up. Each block then receives inputs at a different scale, which makes training fragile.

**RMSNorm** rescales a vector to unit root-mean-square: `x / sqrt(mean(x²) + eps)`. It's LayerNorm without subtracting the mean, which is cheaper and works just as well. Llama, Mistral, Qwen, DeepSeek and most other current LLMs use it.

We apply it **before** each sub-block (*pre-norm*): `x = x + attn(rmsnorm(x))`, then `x = x + mlp(rmsnorm(x))`. The residual stream itself is never normalized, so gradients still flow freely along it. The original 2017 transformer normalized *after* the addition (post-norm), which needed careful warmup to train.

With that, our block is a complete **transformer block**: attention to mix information *across* positions, and an MLP to process *each* position. Stack `n_layer` of them.

(microgpt also normalizes right after the embeddings. As its comment says, it isn't redundant, because of how gradients flow through the residual.)
