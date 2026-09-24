While generating text, a model keeps the keys and values of every past token: the **KV cache** from lesson 07. For long contexts and many users at once, that cache, not the weights, fills GPU memory and limits speed.

**Multi-query attention (MQA)** keeps all the query heads but only **one** key/value head, shared by all of them. **Grouped-query attention (GQA)** is the middle ground: `n_kv_head` groups, each shared by `n_head / n_kv_head` query heads.

The diff is small:
- `attn_wk` and `attn_wv` project to `n_kv_head · head_dim` instead of `n_embd`.
- `k[:, kv_for_head]` repeats each kv head for its group (with `n_head = 4` and `n_kv_head = 2`, the index is `[0, 0, 1, 1]`).

The printout shows the cache shrinking by `n_head / n_kv_head`. Compare the loss curves for `n_kv_head` = 1, 2 and 4: the quality cost is small. Llama 2 70B, Llama 3, Mistral and Qwen all use GQA.

**Going further:** DeepSeek's **MLA** (multi-head latent attention) compresses keys and values into a small shared latent vector instead. It's on the roadmap.
