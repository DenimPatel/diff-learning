Softmax is shift-invariant: adding the same constant to every logit changes nothing. The cross-entropy loss therefore never controls the **absolute level** of the logits, only their differences. The normalizer `log Z = log Σ exp(logits)` can drift, and at large values bf16 rounding (lesson 44) and overflow start to bite. That is one of the causes of the loss spikes that plagued large runs.

### z-loss
PaLM's fix adds a tiny penalty that pins `log Z` near 0:

```
loss = cross_entropy + z_loss_coef · (log Z)²
```

It costs nothing in modeling power, because the model can shift all logits by the same amount. `train_loss_fn` computes `log Z` explicitly (stably, with the max subtracted), and the chart's `log_z` series shows it. PaLM used `1e-4`. The same idea, the **router z-loss**, stabilizes MoE routers (ST-MoE).

### Label smoothing
The cross-entropy target is a one-hot vector: the model is pushed to make the right token's probability exactly 1, which needs infinitely large logits. **Label smoothing** asks for `1 − ε` on the right token and spreads `ε` uniformly. It improves calibration and was standard in the original Transformer and image classifiers. Notice that it makes val *loss* worse: it deliberately stops the model from fitting the true distribution exactly.

### The diff
`train_loss_fn` replaces `loss_fn` for training only. Evaluation keeps the plain cross-entropy, so `train`/`val` stay comparable with lesson 40, and the chart's `loss` is the plain next-token loss too.

### What to try
Compare **High LR, no z-loss** with **High LR, strong z-loss**: with z-loss, `log_z` stays near 0 and val ends lower.

**Up to date:** z-loss is used by PaLM, OLMo 2, Chameleon, Baichuan 2 and many MoE routers, usually at `1e-4`–`1e-5`. Label smoothing is rare in LLM pretraining today, where likelihood is the goal.
