So far the only number we watched was the training loss. That tells us how well the model fits the names it has **already seen**. A model that memorizes them could push it to zero and still be useless on new names. What we actually care about is **generalization**, so we keep a second set of names the model never trains on and measure the loss there.

### The split
The first 10% of the shuffled names become `val_docs`. The model never takes a gradient step on them. The rest (or a `data_fraction` of the rest) become `train_docs`. Each set is tokenized into its own stream, and `get_batch` only ever samples training windows from `train_data`.

### Evaluating
Every `eval_interval` steps, `evaluate()` computes the loss on a **fixed** set of windows from each split, so successive numbers are comparable (a fresh random batch each time would add noise). The chart gets three new series:
- `train`: loss on fixed training windows
- `val`: loss on held-out windows
- `gap`: `val − train`. Near 0 means the model generalizes, and a growing gap means it is **overfitting**.

At the end the val loss is reported in other units. **Bits per character** is `loss / ln 2`, and **perplexity** is `e^loss`, roughly "how many characters the model is choosing between". A uniform guess over 27 tokens has perplexity 27.

### What to try
Run **Memorize 90 names**. With ~90 training names and a wider model, train loss drops toward zero while val loss climbs. Samples marked *copied from the training set* are memorized, not invented. Then run with all the data: the gap stays near zero because this model is far too small to memorize 29k names.

**Up to date:** every result reported from here on is a validation number. LLM papers report held-out perplexity or benchmark scores, and a jump in train/val gap is the first thing to check when an experiment looks too good. Pretraining corpora are now so large that most frontier models see each token roughly once, so overfitting matters most during fine-tuning (Part 4).
