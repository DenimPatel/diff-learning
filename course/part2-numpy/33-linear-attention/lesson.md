Softmax attention costs `O(T²)` and needs a KV cache that grows with every token. Recurrent networks cost `O(T)` with a fixed-size state, but they train one step at a time, which is slow on GPUs. **Linear attention** gets the best of both: it trains in parallel like a transformer and generates like an RNN.

### The layer
Remove the softmax and multiply by a decay that fades with distance:

```
o_t = Σ_{s ≤ t} decay^(t−s) · (q_t · k_s) · v_s
```

In the diff, `decay_mask` replaces the causal mask: lower triangle, with `decay^(t−s)` instead of 1. Every head gets its own decay (0.5, 0.75, 0.88, 0.94), so some heads track the last character and others the whole name, as in RetNet. Without softmax nothing bounds the output's scale, so each head's output goes through RMSNorm.

### The same layer as an RNN
The sum can be rewritten with a running state: a `head_dim × head_dim` matrix per head that accumulates `kᵀv` and decays every step:

```
S_t = decay · S_{t−1} + k_tᵀ v_t        o_t = q_t · S_t
```

`gpt_step` implements the whole model this way, one token in and one token out, in plain NumPy. The script checks that it produces **the same logits** as the parallel version (difference ~1e-15) and then samples with it. Memory stays constant however long the sequence gets.

### The catch
A fixed-size state must *compress* the whole history. Softmax attention can look up any earlier token exactly, and a state cannot. That is why pure linear-attention models struggle with exact recall (copying, retrieving a fact from 100k tokens back).

**Up to date:** modern versions make the decay and gates **input-dependent** (Mamba, Mamba-2, Gated DeltaNet, RWKV-7) so the model chooses what to remember. The current trend is **hybrids**: mostly linear/state-space layers plus a few full-attention layers for recall (Jamba, Qwen3-Next, Kimi Linear, Nemotron-H, MiniMax).
