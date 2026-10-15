Next-token prediction gives one training signal per position. **Multi-token prediction** (MTP) asks the model to also predict tokens further ahead, here the one **after** next, from the same hidden state. Planning further ahead makes the shared trunk learn richer features, and the extra predictions are useful at inference time too.

### The diff
- A new matrix `mtp_proj` turns the final hidden state into a second vector, which goes through RMSNorm and the **shared** `lm_head`. `gpt()` now returns `(logits, mtp_logits)`.
- Batches add `yb2`, the targets shifted by two.
- `loss = main_loss + mtp_weight · mtp_loss`. The chart reports `loss` (the main next-token loss, so it stays comparable with lesson 21) and `mtp`.

DeepSeek-V3's MTP module is a bit richer: it combines the hidden state with the embedding of the true next token and runs a full transformer block before predicting t+2, and it uses `mtp_weight` 0.3 and then 0.1.

### A free draft model
During sampling the script checks how often the MTP head's best guess for the token after next turns out to be the token that is actually sampled. It is right roughly 40% of the time here, against almost never without training. That is a **draft for speculative decoding** (lesson 62) that costs one small extra matmul. DeepSeek-V3 reports 85–90% acceptance for its MTP draft, which nearly doubles decoding speed.

### What to try
Compare the default with **No MTP loss**. On names the main loss barely changes, as expected: Gloeckle et al. find MTP helps mostly for large models and for code, where planning ahead pays off. The inference-time draft benefit appears at every scale.

**Up to date:** DeepSeek-V3/R1, Qwen3-Next, GLM-4.5 and MiMo train with MTP and use the MTP heads for speculative decoding in production (vLLM and SGLang support it).
