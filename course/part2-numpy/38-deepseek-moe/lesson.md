Lesson 25 built a Mixtral-style MoE: a few big experts, top-2 routing, an auxiliary balancing loss. DeepSeek-V3 (and Kimi K2, GLM-4.5, many others since) changed three things.

### 1. Fine-grained experts
Instead of 4 experts with a 128-wide MLP each, use **8 experts with a 32-wide MLP** and route each token to **3**. Compute per token is about the same, but the number of possible expert combinations explodes (C(8,3) = 56 vs C(4,2) = 6), so experts can specialize more narrowly. DeepSeek-V3 has 256 routed experts per layer and uses 8.

### 2. Shared experts
`n_shared` experts process **every** token, with no router. Common knowledge (spelling, grammar, frequent patterns) goes there once, instead of being duplicated in every routed expert. With 3 routed + 1 shared at width 32, each token sees an MLP width of 128, the same as the dense model, while the layer holds 9 experts' worth of parameters.

### 3. Aux-loss-free balancing
The auxiliary loss works, but its gradient competes with the language-modeling loss. DeepSeek's alternative keeps a **bias per expert** that is added to the routing scores **only to choose the top-k**. Gate weights still come from the unbiased scores. After every step, each overloaded expert's bias goes down by `bias_rate` and each underloaded one's goes up:

```python
expert_bias[li] += bias_rate * np.sign(1 / n_expert - frac)
```

No gradient, no extra loss term: a feedback controller. Routing also switches from softmax to independent **sigmoid** scores, normalized over the chosen experts, as in V3.

### What to try
Compare **No balancing** (`busiest` near 0.28) with the default (~0.14, close to the ideal 0.125) and **Classic aux loss**. The language-modeling loss is about the same here. The paper's point is that at scale the bias method balances *without* hurting the loss.

**Up to date:** fine-grained + shared experts with bias balancing is the default open MoE design in 2025–26 (DeepSeek-V3/R1, Kimi K2, GLM-4.5/5). Qwen3 drops the shared expert, and gpt-oss uses fewer, larger experts.
