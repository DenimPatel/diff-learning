This is the first real language model, and the whole pipeline is here: **parameters → forward → loss → backward → update → sample.** Everything after this lesson improves the model *inside* `gpt()`; the rest of the file barely changes.

### Parameters
`state_dict` holds matrices of small random `Value`s:
- `wte` (token embedding): one row of `n_embd` numbers per token. This is the model's "meaning" vector for each character.
- `lm_head`: maps a vector back to one score (a **logit**) per possible next token.

### Forward
`gpt(token_id)` looks up the embedding row and multiplies it by `lm_head`. The result is 27 logits. It only ever sees the *current* character, so it is a **bigram** model: it can learn "after `q` comes `u`", but nothing longer.

### Loss
`softmax` turns logits into probabilities (exponentiate, then normalize; subtracting the max first is only for numerical safety). The loss is the **cross-entropy**, `-log p(correct next token)`, averaged over the name. A uniform guess over 27 tokens gives `ln 27 ≈ 3.30`, which is where the curve starts. Lower is better.

### Update & sampling
The update is plain **SGD**, exactly lesson 03 applied to every parameter. After training we **sample**: start from BOS, draw the next token from the predicted probabilities, feed it back in, and stop when the model picks BOS again.

Watch the loss fall from ~3.3 to ~2.5. The names come out *name-ish*, but they're weird, because the model has no memory beyond one character.
