The model has two tables that relate tokens to vectors. `wte` maps a token **to** a vector at the input, and `lm_head` maps the final vector back **to** a score per token. Both encode "what does this token mean", so why learn them separately? **Weight tying** uses one matrix for both: `logits = x @ wteᵀ`. A token's score is the dot product between the final hidden state and that token's own embedding.

### The diff
`lm_head` disappears from `state_dict` when `tie_weights` is on, and the forward pass uses `wte.transpose(1, 0)` instead. Gradients from the output layer and from the input lookup now both flow into the same table. Our autograd adds them automatically, since `wte` is used twice in the graph.

### Why it matters
Here it saves `n_embd × vocab_size = 864` parameters (3%). In a real model with a 150k-token vocabulary and `n_embd = 4096`, each table is 600M parameters: a large share of a small model. Tying also gives rare tokens a better output vector, because the input side trains it too.

### What to try
Compare tied and **Untied** at the default size (single-batch losses are noisy, so look at the smoothed curves), then at `n_embd = 8`, where one table serving both roles can become a bottleneck.

**Up to date:** small models usually tie (GPT-2, Gemma, Llama 3.2 1B/3B, Qwen 0.5–4B, SmolLM, MobileLLM), because the embedding is a large fraction of their parameters. Large models usually *untie* (Llama 3 8B+, DeepSeek-V3), since the cost is negligible at their size and separate tables are slightly better.
