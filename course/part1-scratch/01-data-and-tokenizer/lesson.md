A language model learns one thing: **given the tokens so far, predict the next token.** Everything we build serves that task, so we start with the data.

### Dataset
The dataset is 32,033 names, one per line, and each name is a *document*. We shuffle them with a fixed seed so every run sees the same order. That makes experiments comparable: when you change a knob, the only thing that changes is the knob.

### Tokenizer
Neural networks work on numbers, not strings. The simplest tokenizer maps each **unique character** to an integer id: `a → 0`, `b → 1`, and so on. On top of that we add one special token, **BOS** (beginning of sequence). We wrap every name in it, `BOS e m m a BOS`, so the model can learn both *how names start* (what follows the first BOS) and *when they end* (predicting BOS).

Real LLMs use subword tokenizers such as BPE, with vocabularies of 100k+ tokens, but the idea is the same: text in, integers out, and a way back.

### Try it
The demo prints the prediction tasks hidden inside one name. A name with 5 letters gives **6 training examples**, one per arrow. Use the `doc_index` knob to look at other names.
