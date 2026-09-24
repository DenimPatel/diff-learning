Compare this file with **lesson 10** (*compare with → 10*): it is the same model, the same parameter names and the same Adam. Three things change.

### 1. Batching
microgpt trained on one name per step. Now we cut the whole dataset into one token stream (`BOS emma BOS olivia BOS ...`) and take a **batch** of `batch_size` random windows of `block_size` tokens. The targets are the same windows shifted by one. One step now averages over 512 predictions, so the gradient is far less noisy.

### 2. All positions at once
Instead of looping over positions and appending to `keys`, we compute `q`, `k` and `v` for **all** positions with one matmul, reshape them to `(B, n_head, T, head_dim)`, and get every attention score with `q @ kᵀ`. Position `t` must still not see the future, so we add a **causal mask**: `-1e9` above the diagonal, which becomes a weight of 0 after the softmax. In lesson 07 we got this for free from the loop.

### 3. log_softmax
Cross-entropy is computed from `log_softmax` (`x - max - log Σ exp(...)`) instead of `log(softmax(x))`, which can underflow to `log(0)`.

Also `n_layer = 2, n_embd = 32`: ~27k parameters trained for 300 steps in seconds, versus 4k parameters taking minutes in Part 1. **This lesson is the baseline:** every branch below changes one thing, and you can overlay their loss curves on this one.
