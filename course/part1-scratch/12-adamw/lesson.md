**Weight decay** pulls every weight slightly toward zero on each step. Big weights must "earn their size" through the gradient, which regularizes the model and keeps its scale under control during long training runs.

With plain SGD, weight decay is the same as adding `λ·w²` to the loss. Adam breaks that equivalence: an L2 term in the loss would get divided by `sqrt(v)` along with everything else, so parameters with large gradients would be barely decayed. **AdamW** *decouples* it by applying `w -= lr · λ · w` directly, outside the adaptive update. That one line is the whole diff.

The chart gets a `wnorm` series, the total size of all the weights. Run with `weight_decay = 0` and `1.0` and compare. On a run this short the loss barely changes, but the weight norm clearly does.

In practice LLMs use `λ ≈ 0.1` and often skip decay on embeddings and norm gains. We decay everything to keep the code short.
