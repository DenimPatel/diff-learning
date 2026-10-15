The model outputs a probability distribution. **Decoding** is the rule that turns it into text, and the same weights can be dull, creative or broken depending on that rule. Part 1 had temperature and top-k/top-p (lesson 13). This lesson adds the rest of the standard toolbox, built on the KV cache.

### The strategies
- **sample**: draw from `softmax(logits / temperature)`. Varied, but at high temperature an unlikely token now and then derails the text.
- **greedy**: always take the most likely token. Deterministic and safe, but it repeats itself, and the most likely sequence is often bland.
- **min-p**: drop every token whose probability is below `min_p × p_max`, then sample. When the model is confident the cut-off is strict, and when many tokens are plausible it keeps them all. That makes it more robust to high temperature than top-p.
- **beam**: keep the `beam_width` best partial sequences at each step and return the best complete ones. It finds higher-probability sequences than greedy, which helps translation and other tasks with one right answer. Ranking by total log-probability favors short outputs, so it is divided by `length^length_penalty`.
- **repetition penalty** (applies to all of them): logits of tokens already in the output are divided by the penalty (or multiplied, if negative).

`beam_search` copies the cache with `list(cache)`. That is cheap because the arrays inside are never modified, only replaced, so beams share the history they have in common.

### The table
For each strategy the script reports how many outputs are **unique**, how many are **real names** from the dataset (memorized or just very typical), and the average length. The trade-off is visible: greedy is 100% "real" with no variety, sampling is varied, and min-p sits in between.

**Up to date:** chat models typically sample with temperature 0.6–1.0 plus top-p or min-p. Reasoning models are often run with specific recommended settings because greedy decoding makes them loop. Beam search is mostly gone from open-ended generation but still used for translation, speech recognition and constrained outputs. Frequency and presence penalties (OpenAI API) are additive versions of the repetition penalty.
