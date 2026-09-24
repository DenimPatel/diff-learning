# Roadmap

Lessons we want next. Each is a small branch off an existing lesson; see [CONTRIBUTING.md](CONTRIBUTING.md).

## Attention
- [ ] **Sliding-window attention** (Mistral, Gemma): a banded causal mask
- [ ] **Multi-head latent attention (MLA)** (DeepSeek-V2/V3): compress K/V into a latent
- [ ] **QK-norm** (OLMo 2, Gemma 3, Qwen3): RMSNorm on queries and keys
- [ ] **Attention sinks / softmax-off-by-one**
- [ ] **KV-cache inference** for the tensor GPT (Part 1 already has it implicitly)
- [ ] **Differential attention**
- [ ] **Linear attention / state-space layers** (Mamba-style) as a contrast

## Architecture
- [ ] Weight tying (`lm_head = wteᵀ`)
- [ ] Dropout
- [ ] LayerNorm vs RMSNorm, post-norm vs pre-norm experiment
- [ ] GELU and other activations
- [ ] MoE: shared expert + fine-grained experts (DeepSeekMoE), aux-loss-free bias balancing
- [ ] Multi-token prediction
- [ ] Logit soft-capping

## Training
- [ ] Train/validation split and overfitting
- [ ] WSD (warmup–stable–decay) schedule
- [ ] Gradient accumulation
- [ ] μP / hyperparameter transfer
- [ ] Mixed precision (simulate bf16 rounding in NumPy)
- [ ] Lion, Adafactor, Sophia and other optimizer comparisons

## Tokenization & data
- [ ] Byte-pair encoding from scratch
- [ ] Training on a different dataset (e.g. tiny Shakespeare)

## Beyond pretraining
- [ ] Supervised fine-tuning on a toy instruction format
- [ ] LoRA
- [ ] DPO on toy preferences
- [ ] Speculative decoding

## Site
- [ ] Share a playground snippet via URL
- [ ] Side-by-side diff mode
- [ ] Quizzes / "predict the curve before you run" prompts
- [ ] Pyodide interrupt support (needs cross-origin isolation)
