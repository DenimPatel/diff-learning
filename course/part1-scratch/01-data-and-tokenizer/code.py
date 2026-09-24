"""
Build a GPT from an empty file, one diff at a time.
Pure, dependency-free Python, following @karpathy's microgpt.py:
https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
"""

import os       # os.path.exists
import random   # random.seed, random.choices, random.gauss, random.shuffle
random.seed(42) # Let there be order among chaos

# Let there be a Dataset `docs`: list[str] of documents (e.g. a list of names)
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
uchars = sorted(set(''.join(docs))) # unique characters in the dataset become token ids 0..n-1
BOS = len(uchars) # token id for a special Beginning of Sequence (BOS) token
vocab_size = len(uchars) + 1 # total number of unique tokens, +1 is for BOS
print(f"vocab size: {vocab_size}")

# --- demo: tokenize one document and look at the next-token prediction task it defines ---
doc_index = 0 # @param {"min": 0, "max": 100} which document to tokenize
doc = docs[doc_index]
tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
print(f"doc: {doc!r}")
print(f"tokens: {tokens}")
print(f"decoded: {''.join(uchars[t] for t in tokens if t != BOS)!r}")
print("the model will learn: given the tokens so far, predict the next one")
for i in range(len(tokens) - 1):
    context = ''.join('.' if t == BOS else uchars[t] for t in tokens[:i + 1])
    target = '.' if tokens[i + 1] == BOS else uchars[tokens[i + 1]]
    print(f"  {context:>16} -> {target}")
