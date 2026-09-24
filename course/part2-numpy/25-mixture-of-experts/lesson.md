A dense model runs every parameter on every token. A **Mixture of Experts** replaces the MLP with `n_expert` separate MLPs (*experts*) plus a tiny **router**. The router scores the experts for each token, and only the **top-k** actually run. Total parameters grow by `n_expert×`, while compute per token grows only by `top_k×`. Mixtral, DeepSeek-V3, Qwen3-MoE and (reportedly) most frontier models are MoEs.

### Routing
`router_probs = softmax(x @ router)` gives each token a distribution over experts. We keep the top-k as a constant 0/1 `mask` (picking an index is not differentiable), then renormalize. The output is `Σ gate_e · expert_e(x)`. The router still learns: gradients reach it through the gate values of the experts it chose.

### Load balancing
Left alone, routing collapses: a slightly better expert gets more tokens, trains more, gets even better, and the rest go unused. The Switch Transformer **auxiliary loss** `n_expert · Σ_e fraction_e · mean_prob_e` is lowest (1.0) when tokens are spread evenly. The chart shows it as `aux`, and `busiest` is the share of tokens taken by the most-loaded expert (`1/n_expert` is perfect). Run **No balancing** and compare.

### Honest simplification
For clarity every expert processes every token and unchosen experts are multiplied by 0. The math is identical, but real implementations *dispatch* each token only to its experts, and that is where the savings come from.

**Up to date:** DeepSeek-V3 uses many small *fine-grained* experts plus an always-on *shared* expert, and balances load with a per-expert bias instead of an aux loss. Both are good Playground exercises starting from this file.
