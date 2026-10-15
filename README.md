# diff-learning

**Build a GPT from an empty file, one diff at a time, and run every step in your browser.**

Inspired by Karpathy's [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95), a complete GPT in about 200 lines of dependency-free Python. This course starts from an empty file and adds **one concept per lesson** (data, autograd, loss, optimizer, attention, …). Each lesson shows exactly the **diff** that concept needs. From there it branches into the tricks of the trade behind modern LLMs:

- **Architecture**: RoPE, GQA, MLA, sliding windows, QK-norm, attention sinks, SwiGLU, mixture of experts, linear attention
- **Training**: validation, dropout, WSD schedules, mixed precision, μP, z-loss, Muon and other optimizers, BPE
- **Post-training**: checkpoints, SFT, LoRA, QLoRA, quantization, distillation, DPO, GRPO
- **Inference**: KV cache, decoding strategies, speculative decoding, constrained decoding

- 📖 **Read** a short explanation.
- ± **Study the diff** against the previous lesson, or compare any two lessons (e.g. base GPT vs MoE).
- ▶ **Run it**. Python runs in the browser via [Pyodide](https://pyodide.org), with nothing to install.
- 🎛 **Turn the knobs.** Every `# @param` line becomes a slider, and loss curves from different runs overlay on one chart.
- ✏️ **Playground.** Edit any lesson freely and see your own diff.

It is a static site with no backend, so it hosts on GitHub Pages.

## Curriculum

**Part 1 · From scratch** (pure Python, every number is a `Value`)

| # | Lesson | Adds |
|---|---|---|
| 00 | An empty file | the plan |
| 01 | Data & tokenizer | names dataset, char tokenizer, BOS |
| 02 | Autograd | `Value`, chain rule, `backward()` |
| 03 | Gradient descent | the training loop, learning rate |
| 04 | Bigram model | embeddings, softmax, cross-entropy, sampling |
| 05 | Adam | momentum + adaptive step sizes |
| 06 | Position, MLP, residuals | `wpe`, ReLU MLP, residual stream |
| 07 | Self-attention | queries/keys/values, causal by construction |
| 08 | Multi-head attention | heads |
| 09 | RMSNorm | pre-norm transformer block |
| 10 | **microgpt** | LR decay, temperature (≈ the gist, line for line) |
| 11 ↳ | Warmup + cosine | LR schedule used by most LLMs |
| 12 ↳ | AdamW | decoupled weight decay |
| 13 ↳ | Top-k / top-p | nucleus sampling |

**Part 2 · Tensors & modern variants** (NumPy, trains in seconds)

| # | Lesson | Adds |
|---|---|---|
| 20 | Tensor autograd | `Value` → `Tensor` (branches off lesson 02) |
| 21 | **microgpt on tensors** | batching, causal mask, log-softmax |
| 22 ↳ | RoPE | rotary position embeddings |
| 23 ↳ | GQA / MQA | shared key/value heads, smaller KV cache |
| 24 ↳ | SwiGLU | gated MLP |
| 25 ↳ | Mixture of Experts | top-k router, load-balancing loss |
| 26 ↳ | Muon | Newton–Schulz orthogonalized updates |
| 27 ↳ | Gradient clipping | global-norm clipping |
| 28 ↳ | Sliding-window attention | banded mask, local + global layers |
| 29 ↳ | QK-norm | RMSNorm on queries and keys, bounded attention logits |
| 30 ↳ | Attention sinks | learned sink logit, softmax off by one |
| 31 ↳ | Multi-head latent attention | low-rank K/V latent, 8x smaller KV cache |
| 32 ↳ | Differential attention | difference of two softmax maps |
| 33 ↳ | Linear attention | decayed linear attention, run as an RNN with a fixed-size state |
| 34 ↳ | Weight tying | `lm_head = wteᵀ` |
| 35 ↳ | LayerNorm, GELU, post-norm | the 2017/GPT-2 choices vs the modern ones |
| 36 ↳ | Logit soft-capping | `cap·tanh(x/cap)` on attention and output logits |
| 37 ↳ | Multi-token prediction | an extra head for token t+2, a free speculative draft |
| 38 ↳ | DeepSeekMoE | fine-grained + shared experts, aux-loss-free bias balancing (branches off 25) |

**Part 3 · Training like the pros** (NumPy, branches off lesson 21)

| # | Lesson | Adds |
|---|---|---|
| 40 | **Train/validation split** | held-out names, val loss, overfitting, bits per character |
| 41 ↳ | Dropout | train/eval modes, regularization on scarce data |
| 42 ↳ | WSD schedule | warmup–stable–decay vs cosine, linear, constant |
| 43 ↳ | Gradient accumulation | micro-batches, same gradient as one big batch |
| 44 ↳ | Mixed precision | simulated bf16/fp16, master weights, loss scaling |
| 45 ↳ | Initialization and μP | fan-in / GPT-2 init, learning-rate transfer across widths |
| 46 ↳ | z-loss and label smoothing | keeping the output layer in check |
| 47 ↳ | Optimizer zoo | Lion, Adafactor, SGD vs Adam, optimizer memory |
| 48 ↳ | Byte-pair encoding | a learned subword tokenizer, fair comparison in bits per character |

**Part 4 · Beyond pretraining** (every lesson starts from a cached pretrained checkpoint)

| # | Lesson | Adds |
|---|---|---|
| 50 | **Checkpoints** | save/load weights + optimizer state, resume after a crash, reuse a base model |
| 51 ↳ | Supervised fine-tuning | a new special token, loss masking, catastrophic forgetting |
| 52 ↳ | LoRA | frozen base, low-rank adapters, detach and merge |
| 53 ↳ | Quantization | int8/int4/int2, per-tensor vs group-wise scales, outliers |
| 54 ↳ | QLoRA | NF4 4-bit base + LoRA (branches off 52) |
| 55 ↳ | Knowledge distillation | a 6x smaller student learns from soft targets |
| 56 ↳ | DPO | preference pairs, frozen reference, β, reward hacking |
| 57 ↳ | GRPO | RL with verifiable rewards: group advantages, PPO clipping, KL |

**Part 5 · Inference** (NumPy, branches off lesson 21)

| # | Lesson | Adds |
|---|---|---|
| 60 | **KV cache** | incremental decoding, O(T) instead of O(T²) |
| 61 ↳ | Decoding strategies | greedy, min-p, beam search, repetition penalty |
| 62 ↳ | Speculative decoding | n-gram draft, one-pass verification, lossless rejection sampling |
| 63 ↳ | Constrained decoding | token masks that guarantee a format (JSON mode) |

See [ROADMAP.md](ROADMAP.md) for what's next and [CONTRIBUTING.md](CONTRIBUTING.md) to add a lesson.

## Run locally

```bash
pip install -r requirements-dev.txt   # numpy + pytest; the site itself needs nothing
python tools/build.py                 # site/ + course/ -> dist/ (with course.json)
python -m http.server -d dist 8000    # open http://localhost:8000
pytest -q                             # every lesson runs, knobs parse, tree is valid (~2 minutes)
```

Every lesson is also a plain script: `cd /tmp && python /path/to/course/part1-scratch/10-microgpt/code.py`.

## How it works

```
course/<part>/<NN-slug>/
  code.py       full code snapshot after this lesson (runnable on its own)
  lesson.md     the explanation
  lesson.toml   title, parent, summary, tags, references, experiments (knob presets)
tools/          build.py bundles course/ into dist/course.json; params.py parses # @param knobs
site/           static frontend: index.html, app.js, worker.js (Pyodide), params.js, styles.css
tests/          pytest: runs every lesson and every experiment with tiny step counts
```

- **Diffs are never stored.** Each lesson keeps its *full* code and names its `parent`, and the site computes `parent → lesson` diffs in the browser. Editing an early lesson therefore updates every diff after it, and new variants are just new directories.
- **Knobs** are Colab-style annotations that keep the file valid Python:
  `learning_rate = 0.01  # @param {"min": 0.0001, "max": 1, "log": true} step size`
- **Charts** come from stdout: any printed line like `step 12 / 300 | loss 2.31 | lr 0.01` becomes live series.

## Deploying to GitHub Pages

`.github/workflows/pages.yml` runs the tests and builds `dist/` on every push and PR, and deploys pushes to `main`. Enable it once under **Settings → Pages → Build and deployment → Source: GitHub Actions**.

## Credits

The Part 1 destination is Andrej Karpathy's [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) (also see [micrograd](https://github.com/karpathy/micrograd), [makemore](https://github.com/karpathy/makemore), [nanoGPT](https://github.com/karpathy/nanoGPT)). The dataset is makemore's `names.txt`.
