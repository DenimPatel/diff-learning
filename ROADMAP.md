# Roadmap

Lessons we want next. Each is a small branch off an existing lesson; see [CONTRIBUTING.md](CONTRIBUTING.md).

## Attention
- [x] **Sliding-window attention** (Mistral, Gemma): a banded causal mask (lesson 28)
- [x] **Multi-head latent attention (MLA)** (DeepSeek-V2/V3): compress K/V into a latent (lesson 31)
- [x] **QK-norm** (OLMo 2, Gemma 3, Qwen3): RMSNorm on queries and keys (lesson 29)
- [x] **Attention sinks / softmax-off-by-one** (lesson 30)
- [x] **KV-cache inference** for the tensor GPT (lesson 60)
- [x] **Differential attention** (lesson 32)
- [x] **Linear attention / state-space layers** as a contrast (lesson 33)
- [ ] Gated attention (output gate, Qwen3-Next)
- [ ] Input-dependent decays (Mamba-2 / Gated DeltaNet) and a hybrid linear + full attention model
- [ ] Partial RoPE, NoPE layers, RoPE scaling for longer context (YaRN)
- [ ] FlashAttention-style tiled attention with an online softmax

## Architecture
- [x] Weight tying (`lm_head = wteᵀ`) (lesson 34)
- [x] Dropout (lesson 41)
- [x] LayerNorm vs RMSNorm, post-norm vs pre-norm experiment (lesson 35)
- [x] GELU and other activations (lesson 35)
- [x] MoE: shared expert + fine-grained experts (DeepSeekMoE), aux-loss-free bias balancing (lesson 38)
- [x] Multi-token prediction (lesson 37)
- [x] Logit soft-capping (lesson 36)
- [ ] Value residual learning / layer-wise skip connections (modded-nanogpt)

## Training
- [x] Train/validation split and overfitting (lesson 40)
- [x] WSD (warmup–stable–decay) schedule (lesson 42)
- [x] Gradient accumulation (lesson 43)
- [x] μP / hyperparameter transfer (lesson 45)
- [x] Mixed precision (simulate bf16 rounding in NumPy) (lesson 44)
- [x] Lion, Adafactor and other optimizer comparisons (lesson 47)
- [x] z-loss and label smoothing (lesson 46)
- [ ] Sophia, SOAP and Shampoo
- [ ] FP8 training with per-block scales
- [ ] Data parallelism / ZeRO simulated across "devices"
- [ ] Scaling laws: fit loss vs model size and data from a sweep of small runs

## Tokenization & data
- [x] Byte-pair encoding from scratch (lesson 48)
- [ ] Training on a different dataset (e.g. tiny Shakespeare)
- [ ] Data mixing and curriculum / annealing on high-quality data

## Beyond pretraining
- [x] Checkpoints and a reusable base model (lesson 50)
- [x] Supervised fine-tuning on a toy instruction format (lesson 51)
- [x] LoRA (lesson 52) and QLoRA (lesson 54)
- [x] Post-training quantization (lesson 53)
- [x] Knowledge distillation (lesson 55)
- [x] DPO on toy preferences (lesson 56)
- [x] GRPO / RL with verifiable rewards (lesson 57)
- [x] Speculative decoding (lesson 62)
- [x] Decoding strategies and constrained decoding (lessons 61, 63)
- [ ] Reward model + PPO (classic RLHF)
- [ ] Chat template with multi-turn masking
- [ ] Tool calling with a toy calculator
- [ ] Model merging (task arithmetic, TIES)
- [ ] Long-context extension by fine-tuning with scaled RoPE

## Site
- [ ] Share a playground snippet via URL
- [x] Side-by-side diff mode (with word-level highlights)
- [ ] Quizzes / "predict the curve before you run" prompts
- [ ] Pyodide interrupt support (needs cross-origin isolation)
