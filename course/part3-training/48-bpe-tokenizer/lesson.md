So far every character has been one token. Real LLMs use **subword tokens**: common chunks like `ing`, ` the` or `ella` get their own ID. Text then needs fewer tokens, the context window covers more of it, and the model spends less effort spelling out common words. The standard way to build such a vocabulary is **byte-pair encoding**.

### Learning the merges
1. Start with single characters.
2. Count every adjacent pair of tokens in the training data (`pair_counts`).
3. Merge the most frequent pair into a new token everywhere (`merges[best] = new id`).
4. Repeat `num_merges` times.

The first merges for names are `an`, `ar`, `el`, `en`, …, and later ones build on earlier ones (`ali`, `yah`). Counting over *distinct* names with their frequencies, instead of the whole stream, keeps this fast.

### Tokenizing
`tokenize()` applies the merges to a new string in the **order they were learned**: always merge the adjacent pair with the lowest rank first. That reproduces exactly how the training data was segmented. Decoding just concatenates each token's string.

### Comparing tokenizers fairly
Loss per token is **not comparable** across tokenizers: a token that stands for 3 characters carries more information and is harder to predict. The fair metric is **bits per character**: `val_bpc = loss per token / characters per token / ln 2`. Switch the chart to `val_bpc`. With 50 merges it ends lower than with characters (3.21 vs 3.26), and with 150 lower still, even though `loss` per token goes up.

### What to try
Compare **No merges** and **150 merges** on `val_bpc`, and look at the merges printed at the top.

**Up to date:** GPT-4o and Llama 3 use byte-level BPE with 128k–200k tokens (bytes rather than characters, so any text can be encoded). Tokenization explains many odd LLM failures, such as counting letters or doing arithmetic on numbers split into strange chunks. Byte-level, tokenizer-free models (Byte Latent Transformer, H-Net) are an active research direction.
