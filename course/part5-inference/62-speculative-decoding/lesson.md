Each new token needs one full forward pass of the model, and for a large model that pass is limited by **reading every weight from memory**, not by arithmetic. Scoring 5 tokens in one pass costs about the same as generating 1. **Speculative decoding** uses that spare compute: a cheap **draft** guesses the next `k` tokens, the **target** model checks all of them in one pass, and every correct guess is a token we got almost for free.

### The draft
Here the draft is an n-gram table: counts of which character follows the last 1 (`bigram`) or 2 (`trigram`) characters in the data. It needs no training and costs almost nothing. Real systems use a small model from the same family, extra prediction heads (Medusa, EAGLE, DeepSeek's MTP), or n-gram lookups in the prompt, which is exactly this idea and works well for code editing and RAG.

### One round (`generate_speculative`)
1. The draft samples `k` guesses `g₁…g_k`, remembering its probabilities `q`.
2. The target runs once on `[last token, g₁ … g_k]` (with the KV cache) and returns `k + 1` next-token distributions `p`.
3. Each guess is **accepted with probability `min(1, p(g)/q(g))`**, in order. At the first rejection, a replacement is sampled from the leftover `max(0, p − q)` (renormalized) and the round ends. If all `k` are accepted, a bonus token comes from `p_{k+1}`. Either way every round produces at least one token.
4. The KV cache is **rolled back** to drop the rejected positions.

The accept/reject rule is rejection sampling: the tokens come out distributed **exactly** like samples from the target alone. Speculative decoding is lossless, unlike quantization or distillation. The script checks this empirically. Plain and speculative samples have the same statistics up to sampling noise.

### What to try
Read `tokens per pass`: with the trigram draft about 2.4 tokens come out of each target pass. Try **Bigram draft** and **Guess 8 ahead**. The speedup depends on the acceptance rate, so it is largest for predictable text (code, structured output, low temperature).

**Up to date:** speculative decoding is built into vLLM, SGLang, TensorRT-LLM and llama.cpp, and is used to serve most frontier models. EAGLE-3 and MTP heads trained with the model reach 3–4 accepted tokens per pass, and *tree* speculation verifies several candidate branches at once.
