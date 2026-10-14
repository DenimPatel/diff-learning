An attention logit is `q · k / √d`. Nothing stops the network from making `q` and `k` longer and longer, and in large models it does. Logits grow into the hundreds, softmax becomes one-hot, its gradient vanishes, and training hits a loss spike or diverges. This **attention-logit growth** was one of the main instabilities found when scaling ViT-22B and LLMs.

### The fix
Normalize them. `q = rmsnorm(q) · gain` and `k = rmsnorm(k) · gain` per head, right before the dot product. Each query and key then has length ≈ `√head_dim × gain`, so the logits are bounded by the learned gains rather than by whatever the weights drifted to. The gains (`attn_q_gain`, `attn_k_gain`, initialized to 1) let the model still learn sharp attention where it needs it.

### The chart
Switch the metric to `max_attn_logit`, the largest attention logit in each forward pass. With QK-norm it stays near 4. Run **Without QK-norm** and it climbs past 20, and at a high learning rate into the hundreds.

### Honest note
At our scale QK-norm does not improve the loss. The instability it prevents shows up in long runs of big models at high learning rates. What you can see here is the *mechanism*: the logits stay bounded. That is also why it pairs well with aggressive optimizers like Muon (lesson 26).

**Up to date:** QK-norm is standard in OLMo 2, Gemma 3, Qwen3, Kimi K2 and most new open models, often together with dropping biases and using a learned scale. Some models (Qwen3.5) use a zero-centered variant. Kimi K2 introduced *QK-clip*, which rescales the query/key weights when logits exceed a threshold.
