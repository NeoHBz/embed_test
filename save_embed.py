from sentence_transformers import SentenceTransformer
import os
import faiss
import numpy as np
import json
import nltk

# Download the sentence tokenizer model (only needs to be done once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

model = SentenceTransformer("all-MiniLM-L6-v2")

# Load file paths
data_dir = "ttt"
data_paths = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".txt")])

# Read contents and filenames, and chunk documents
docs = []
doc_metadatas = []
for path in data_paths:
    content = open(path).read().strip()
    doc_id = os.path.basename(path)
    sentences = nltk.sent_tokenize(content)
    for i, sentence in enumerate(sentences):
        docs.append(sentence)
        doc_metadatas.append({"doc_id": doc_id, "chunk_num": i})


# Embed
embeddings = model.encode(docs)
embeddings = np.array(embeddings).astype("float32")

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save index
faiss.write_index(index, data_dir + "_index.faiss")

# Save metadata
with open(data_dir + "_meta.json", "w") as f:
    json.dump({"doc_metadatas": doc_metadatas, "docs": docs}, f)
