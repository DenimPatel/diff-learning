Softmax attention never gives exactly zero weight to anything. Over long contexts, thousands of irrelevant tokens each get a little, and together that **attention noise** drowns out the few tokens that matter. The Differential Transformer borrows an idea from electronics: a differential amplifier subtracts two signals so that the noise they share cancels.

### The diff
Each head's query and key are split into two halves, giving two attention maps:

```
attn = softmax(q₁ k₁ᵀ / √d) − λ · softmax(q₂ k₂ᵀ / √d)
```

Both maps put some background weight on irrelevant tokens, and the subtraction removes it. What survives is the attention that the first map gives *more* than the second. `λ` is learned per head (`attn_lambda`), initialized with the paper's depth-dependent schedule `0.8 − 0.6·e^(−0.3·layer)`. Weights can now be negative and rows sum to `1 − λ`, so each head's output gets its own RMSNorm and is rescaled by `1 − λ_init`. The chart's `lambda` series shows how the learned values move.

### What to expect
With 16-character names there isn't much noise to cancel, so the loss matches lesson 21. The paper reports its gains (better long-context retrieval, fewer hallucinations, fewer activation outliers) at billions of parameters and long contexts. Treat this lesson as a working model of the idea rather than a demonstration of the benefit.

**Up to date:** differential attention is a research architecture that has not (yet) been adopted by the major open models. Its goal, attention that can ignore tokens, is shared by attention sinks and gated attention (lesson 30), which are in production.
