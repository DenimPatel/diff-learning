Full attention compares every position with every earlier one: `O(T²)` compute, and a KV cache that grows with every token. For long documents most of that attention goes to nearby tokens anyway. **Sliding-window attention** lets position `t` attend only to positions `t − window + 1 … t`.

### The diff
It's only the mask. `local_mask` is the causal triangle with everything older than `window` positions also set to `−1e9`, so the attention matrix becomes a diagonal **band**. Nothing else changes. In a real implementation the band also skips the computation, and the KV cache becomes a rolling buffer of `window` entries (Mistral 7B).

### Depth extends the reach
A local layer only sees `window` positions, but the next layer attends to positions that have *already* mixed in their own windows. After `L` layers the top layer's receptive field is about `L × (window − 1) + 1` positions, like a stack of convolutions. The script prints it.

### Local + global
Some facts really are far away. Current models therefore **interleave** local layers with a few full-attention (global) layers: Gemma 2 alternates 1:1, Gemma 3 uses 5 local per global, gpt-oss alternates. `global_every` does the same here. The global layers need a full cache, and the local ones only `window` entries.

### What to try
**Window 1** removes attention entirely, which shows how much it was doing. Compare `window` 2, 4 and 16. Names are short, so a small window loses little. That is exactly the bet these models make about most text.

**Up to date:** sliding windows (often 4k tokens) interleaved with global layers are standard in Gemma 2/3, gpt-oss, Mistral's earlier models and many 2025–26 open models. Ai2's OlmPool experiments find that stacking SWA on top of other KV-saving tricks like GQA can hurt long-context quality more than either alone. Combine with care.
