Real training runs last days or months, and machines fail. Everything after pretraining (fine-tuning, quantization, preference training) also starts from a model someone **already trained**. Both need a **checkpoint**: the training state written to a file and read back later.

### What goes in a checkpoint
- The **weights**: one array per parameter, by name.
- The **optimizer state**: Adam's `m` and `v`. Without them a resumed run restarts with empty moments, and the bias correction thinks it is at step 1, so the first updates after a resume are badly scaled.
- The **step**, so that the learning-rate schedule picks up where it left off.

`save_checkpoint` writes all three with `np.savez`. `load_checkpoint` copies them back *into* the existing tensors, so `params`, `m` and `v` keep pointing at the same objects. For an exact resume you would also save the random-number generator's state. Here the batches after a resume differ slightly.

### Naming it after its config
`ckpt_path` is a hash of every setting that changes the result: architecture, optimizer, steps, batch, data and the parameter shapes. A run with the same settings finds the file and skips straight to the end. A run with different settings gets a new file and trains from scratch. In the browser the file lives in Pyodide's in-memory file system, which survives between runs until you press **Stop** or reload the page.

The training loop is now called **pretraining** (`pretrain_steps`). In the next lessons it runs once, gets cached, and every Part 4 lesson fine-tunes the same base model. The sampling loop also became a small `generate()` helper that we reuse.

### What to try
Run **Crash at step 150**, then run again with the default knobs. The second run prints `loaded …: resuming at step 100` and finishes the remaining 200 steps. Run a third time and it skips training entirely.

**Up to date:** open-weight models ship as `safetensors` files, which hold the weights only, under the same `layer0.attn_wq`-style names. Training frameworks such as PyTorch DCP and Megatron additionally shard the optimizer state across GPUs and save it asynchronously, so that a save doesn't pause training.
