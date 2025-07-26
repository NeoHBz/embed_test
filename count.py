import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize
import os

docs_dir = "ttt"
paths = [os.path.join(docs_dir, f) for f in os.listdir(docs_dir) if f.endswith(".txt")]

total = 0
for path in paths:
    content = open(path).read()
    total += len(sent_tokenize(content))

print("Estimated total vectors:", total)
