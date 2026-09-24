A single attention head produces *one* weighted average per position, so it has to pick what to focus on. With **multiple heads**, one can track "the previous letter" while another tracks "is this the start of the name".

The implementation is almost free. Split `q`, `k` and `v` into `n_head` slices of `head_dim = n_embd / n_head`, run the same attention in each slice independently, and concatenate the results. The parameter count doesn't change at all: `wq/wk/wv/wo` are still `n_embd × n_embd`. Only the scaling becomes `sqrt(head_dim)`.

Try the `n_head` knob: `1` is exactly lesson 07, `4` is microgpt's choice, and `16` gives heads that are a single number wide.
