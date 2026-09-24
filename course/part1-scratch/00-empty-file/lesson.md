Every lesson in this course is **one diff**. We start from an empty file, and by lesson 10 it has grown into a GPT that trains on 32,000 names and makes up new ones. That GPT is essentially [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95): about 200 lines of plain Python that contain *the complete algorithm*. As Karpathy says, "everything else is just efficiency".

### How to use this page

- **Read** the short explanation of the concept.
- **Study the diff.** Green lines are added, red lines are removed. It is exactly the code needed for the new idea, and nothing else. Use *compare with* to diff against any other lesson (e.g. the finished GPT vs. mixture of experts).
- **Run it.** The code runs *in your browser* ([Pyodide](https://pyodide.org)); nothing is installed and nothing leaves your machine.
- **Turn the knobs.** Every line tagged `# @param` becomes a slider. Change a value, run again, and the loss curves overlay so you can compare.
- **Break it in the Playground.** Edit the code freely and see your own diff against the lesson.

### The map

**Part 1 · From scratch** is pure Python with no dependencies. Every number is a `Value` object, so it is slow, and that is on purpose: you can see everything.

`data → autograd → gradient descent → bigram model → Adam → MLP → attention → multi-head → RMSNorm → microgpt`

Side branches off the finished GPT: learning rate warmup and cosine decay, AdamW, top-k/top-p sampling.

**Part 2 · Tensors & modern variants** rebuilds autograd on NumPy arrays and trains the same GPT in seconds. Branches off it cover the pieces modern LLMs use: RoPE, GQA, SwiGLU, mixture of experts, Muon, gradient clipping.

The file is empty apart from a docstring. Click the next lesson.
