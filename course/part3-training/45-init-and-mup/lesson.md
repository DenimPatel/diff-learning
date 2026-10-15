microgpt initializes every matrix with `std = 0.08`. That happens to suit `n_embd = 32`, but a fixed number can't be right for every width. Make the model wider and every dot product sums over more terms, so activations grow. Make it deeper and every block adds to the residual stream, which grows too. Getting the scales right is what lets one recipe work at every size.

### Initialization: `init`
- **fixed**: 0.08 everywhere (microgpt).
- **fan_in**: `std = 1/√n_in`, so that `x @ W` has about the same scale as `x` at any width (Glorot/He/LeCun-style). Embeddings use `std = 1`.
- **gpt2**: `std = 0.02`, and the two matrices that write into the residual stream (`attn_wo`, `mlp_fc2`) are scaled by `1/√(2·n_layer)`, so that the stream's variance doesn't grow with depth.

### μP: `parametrization`
Even with good initialization, the **best learning rate changes with width** in the standard parametrization (SP): wider models need smaller learning rates. Tuning a large model then means expensive sweeps *at that size*. **μP** (Maximal Update Parametrization) rescales things so that the optimum stays put:
- hidden matrices (both sides grow with width) get learning rate `lr / width_mult`, because with Adam each entry moves by about `lr`, and n times more entries move the output n times more.
- the output logits are multiplied by `1 / width_mult`.

`width_mult = n_embd / base_width`. At the base width both parametrizations are identical.

### What to try
Run **Wide, standard, LR 0.03** and **Wide, μP, LR 0.03**. A sweep we ran:

| | LR 0.003 | LR 0.01 | LR 0.03 |
|---|---|---|---|
| SP, width 32 | 2.38 | 2.31 | **2.29** |
| SP, width 128 | **2.26** | 2.30 | 2.58 |
| μP, width 128 | 2.29 | **2.22** | 2.23 |

In SP the best learning rate moved by 10×. In μP it stayed, and the wide model is better at the narrow model's best setting. That is **hyperparameter transfer**: tune on a small proxy, train the big model once.

**Up to date:** μP or its descendants are used to tune large runs (Cerebras-GPT, MiniCPM and others). Newer work extends transfer to depth (depth-μP, CompleteP) and to optimizers like Muon, which bake similar width scaling into their update.
