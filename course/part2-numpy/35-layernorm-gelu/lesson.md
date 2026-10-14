microgpt uses the modern defaults: RMSNorm, ReLU (where modern models use a SiLU-gated MLP) and **pre-norm**. The 2017 Transformer and GPT-2 made different choices, and this lesson puts both behind knobs so that you can compare them in the same model.

### Normalization: `norm_type`
- **LayerNorm** subtracts the mean, then divides by the standard deviation (GPT-2, BERT; both also learn a gain and bias, left out here).
- **RMSNorm** only divides by the root mean square. It is cheaper, has no mean to compute, and works just as well, which is why Llama, Qwen, Gemma and DeepSeek use it.

### Activation: `activation`
- **ReLU** `max(0, x)`: the original.
- **GELU** `x · Φ(x)`, a smooth ReLU (GPT-2, BERT). We use its tanh approximation, with `tanh` built from `sigmoid`.
- **SiLU / swish** `x · sigmoid(x)`: the gate inside SwiGLU (lesson 24), used by most current LLMs.

### Where the norm goes: `norm_position`
- **post-norm** (2017): `x = norm(x + block(x))`. The residual stream itself is normalized after every block, so gradients to early layers must pass through every norm. Deep post-norm models need careful warmup and small learning rates.
- **pre-norm** (GPT-2 onwards): `x = x + block(norm(x))`. The residual stream is a clean highway from the output to every layer, and deep models train stably at higher learning rates.

### What to try
Run **Deep post-norm, high LR** and then **Deep pre-norm, high LR**, and overlay them. This gap, much larger in 100-layer models, is why every modern LLM is pre-norm.

**Up to date:** the standard 2026 recipe is pre-norm RMSNorm with a SwiGLU MLP. Some models add extra norms: Gemma 2/3 normalize both before *and* after each block ("sandwich" norm), OLMo 2 moved to a post-norm-inside-the-residual variant for stability, and QK-norm (lesson 29) normalizes inside attention.
