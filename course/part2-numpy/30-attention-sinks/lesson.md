Softmax weights always sum to 1, so every head *must* attend somewhere, even a head whose job does not apply to the current token. Trained LLMs solve this by dumping the unwanted attention on a token that carries no information, usually the first one. These **attention sinks** soak up a large share of attention. They also create huge activations that make quantization hard (lesson 53), and they break naive sliding-window caches: StreamingLLM found that evicting the first token wrecks the model.

### The fix: an explicit sink
Give each softmax an extra option that means "nothing":

```
weights_i = exp(l_i) / (Σ_j exp(l_j) + exp(sink))
```

The weights now sum to *less* than 1, and the missing mass goes nowhere. `softmax_with_sink` implements this (with the usual max-subtraction for stability). Two flavors are available through `sink`:
- **off-by-one**: a fixed `sink = 0`, i.e. `1 +` in the denominator (Evan Miller's *softmax₁*).
- **learned**: one learnable sink logit per head and layer (`attn_sink`), as in gpt-oss.

### The chart
- `sink_mass`: the share of attention that went to the sink, here about 20%. The model uses the option.
- `bos_mass`: the share that went to `BOS` tokens. With **No sink** it rises, as heads fall back on the implicit sink.

The loss barely changes at this size. The payoff is cleaner activations: fewer outliers to quantize and no special token to keep forever in a streaming cache.

**Up to date:** gpt-oss uses learned sinks. Qwen3-Next and others use **gated attention**, a sigmoid gate on the attention output that can zero it out, which removes sinks the other way. Recent work finds sliding-window attention *with sinks* holds up remarkably well on long contexts.
