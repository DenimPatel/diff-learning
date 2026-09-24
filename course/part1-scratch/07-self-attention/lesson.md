Until now each position was processed on its own. **Self-attention** lets a position *look back* at every earlier one and pull in what it needs.

For the current position we compute three vectors from `x`:
- **query** `q`: *what am I looking for?*
- **key** `k`: *what do I contain?* (stored for every position)
- **value** `v`: *what do I pass on if someone attends to me?* (stored for every position)

Then:
1. Score every past position `t`: `q · k_t / sqrt(n_embd)`. Dividing by the square root keeps the scores from growing with the dimension, which would saturate the softmax.
2. `softmax` turns the scores into **attention weights** that sum to 1.
3. The output is the weighted average of the values. `attn_wo` projects it back, and a residual adds it to the stream.

### Causal, for free
Notice `keys[li].append(k)`. We run the sequence **one position at a time** and only append keys and values as we go, so position `t` can only ever see positions `≤ t`. That is exactly the *causal mask* GPTs need, with no mask at all. (In Part 2 we process all positions in parallel and need an explicit mask.) The same lists are also a **KV cache**: during sampling, earlier positions are never recomputed.

Compare the loss to lesson 06 at the same number of steps. The model can now use the letters before the current one.
