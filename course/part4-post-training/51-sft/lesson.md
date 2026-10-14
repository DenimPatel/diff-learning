A pretrained model is a **document completer**: give it `BOS` and it writes a name. It doesn't follow instructions, because it has never seen any. **Supervised fine-tuning** fixes that by continuing training on examples of the behavior we want: a prompt, then the ideal response. Chat models get this from conversations, and here the toy task is *write the name backwards*:

```
BOS e m m a = a m m e BOS
    └ prompt ┘ └ response ┘
```

### A new special token
`=` didn't exist during pretraining, so we add it, the same way chat models add `<|user|>` or `<|im_start|>`. That means one new row in `wte` and one new column in `lm_head`. Initializing them to the **mean** of the existing rows starts the new token as an "average" token rather than random noise. Real fine-tuning libraries do the same when they resize the embeddings.

### Loss masking
`sft_batch` builds a 0/1 `mask` over the targets. With `mask_prompt = true` only the response tokens and the final `BOS` count: the model is graded on its *answer*, not on predicting the user's question. Padding never counts. `sft_loss` is the usual cross-entropy, multiplied by the mask and divided by the number of counted tokens. In TRL this is called *completion-only loss*.

### Training and evaluation
The loop is the same loop with a fresh Adam state and a smaller learning rate. Two new series appear:
- `acc`: how many response tokens it gets right on **held-out** names, so it has to learn the procedure rather than memorize answers.
- `names`: the pretraining validation loss. Watch it climb. That is **catastrophic forgetting**: full fine-tuning overwrites what the model knew, and after SFT it barely writes names anymore.

At the end, `reverse_with_model` decodes greedily (temperature ≈ 0) and reports **exact match** on 50 held-out names.

### What to try
Run **Learn from the prompt too** and look at the last three samples: the model learned to *write* prompts as well as answer them. Compare **Gentle fine-tune** with the default: there is a direct trade-off between learning the task and forgetting. The next lesson, **LoRA**, changes that trade-off by training only a small number of new parameters.

**Up to date:** modern post-training starts with SFT on a few thousand to a few million curated conversations. LIMA showed that 1,000 good examples go a long way. A common remedy for forgetting is to mix some pretraining data into the SFT batches (*replay*). Try it in the Playground: draw half of each batch with `get_batch(train_data, …)`.
