microgpt (Part 1) generated text by appending keys and values to a list, one token at a time. When we moved to tensors (lesson 21) we lost that. To generate each new token, `gpt()` re-ran the **whole sequence so far**, so a sequence of `T` tokens cost `1 + 2 + … + T = O(T²)` position computations. This lesson brings the cache back.

### Why caching is allowed
Attention is causal: position `t` only looks at positions `≤ t`. So the keys and values computed for earlier positions **never change** when new tokens arrive. Compute them once, store them, and each new step only needs the newest token's query, key and value.

### The diff
- `gpt(idx, cache)`: during inference, `cache` is a list with one `(keys, values)` pair per layer, and `idx` holds only the new tokens.
- `past` = how many positions the cache already holds. Position embeddings start at `past`, and the causal mask is the slice `[past:past+T, :past+T]`.
- In each layer the new `k, v` are appended to the cache and attention runs over all of them. Training passes no cache and is unchanged.
- `generate(use_cache)` runs both ways with the same random seed, and checks that the samples are **identical** and how many positions each one processed.

### Prefill and decode
Real servers split generation into two phases. **Prefill** runs the whole prompt in one batched pass and fills the cache: lots of compute, fast. **Decode** then generates one token per step, which is limited by reading the weights and the cache from memory. With our 16-token names Python overhead hides the speedup, but count the positions processed. At 100k tokens of context, the cache is the difference between minutes and milliseconds per token.

### The cost
The cache needs `2 × layers × positions × n_embd` numbers per sequence. For a 70B model at 128k context that is tens of GB *per user*. That is why GQA (lesson 23) and MLA (lesson 31) shrink the K/V heads, why caches get quantized, and why vLLM's PagedAttention manages cache memory like virtual memory pages.

**Up to date:** every LLM server uses a KV cache. Current work is about making it smaller (GQA, MLA, sliding windows, quantized caches), sharing it (prefix caching for common system prompts) and moving it (disaggregated prefill/decode servers).
