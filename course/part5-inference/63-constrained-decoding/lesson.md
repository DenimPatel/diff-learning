Programs that consume model output need it in an exact format: valid JSON, a function call that matches a schema, one label out of a fixed list. Prompting for the format usually works, but "usually" breaks a parser. **Constrained decoding** makes it a guarantee. At every step, every token that would break the format gets probability 0 before sampling.

### The format
A tiny template language stands in for a JSON schema or a grammar. Each symbol is one letter: `C` a consonant, `V` a vowel, `?` any letter, anything else that exact letter. `allowed(position)` returns the set of valid next tokens, and once the template is complete the only valid token is `BOS` (end). Real systems compile a JSON schema or a context-free grammar into an automaton that answers the same question, "which tokens are valid next?", in microseconds.

### The diff
In `generate(constrained=True)`, the logits of forbidden tokens are set to `−∞`. That's it: softmax gives them probability 0 and the rest is renormalized. The script compares this with the naive alternative, **rejection sampling** (generate freely, discard what doesn't fit). For `CVCVC` about 1 in 10 free samples fit, and for `zz?` none of 300 do.

### The catch
The mask is **myopic**. It only removes tokens at the current step and can't plan ahead, so the model can be forced down a path it would never have chosen. With `k????a`, a model that wanted to end the name early ends up writing `keniaa`. Formally, masking samples from a *different* distribution than "the model's distribution conditioned on the output being valid". Recent work also finds that strict formats can hurt reasoning quality, so a common pattern is to let the model think freely and constrain only the final answer.

**Up to date:** OpenAI/Anthropic/Gemini structured outputs, vLLM and SGLang (XGrammar), llama.cpp (GBNF) and Outlines all use this approach. Tokens span several characters, so the hard engineering is computing, for every state of the grammar, which of the ~100k vocabulary tokens are valid, fast enough to add no latency.
