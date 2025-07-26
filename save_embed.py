from sentence_transformers import SentenceTransformer
import os
import faiss
import numpy as np
import json
import nltk
import torch

# Check for GPU availability and enforce GPU usage
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not available. This script requires GPU support.")

device = torch.device("cuda")
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# Download the sentence tokenizer model (only needs to be done once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Load model with GPU enforcement
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

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

# Embed using GPU
print(f"Encoding {len(docs)} documents on GPU...")
embeddings = model.encode(docs, device=device, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

# Build FAISS index with GPU support
res = faiss.StandardGpuResources()
gpu_index = faiss.GpuIndexFlatL2(res, embeddings.shape[1])
gpu_index.add(embeddings)

# Convert back to CPU index for saving
index = faiss.index_gpu_to_cpu(gpu_index)

# Save index
faiss.write_index(index, data_dir + "_index.faiss")

# Save metadata
with open(data_dir + "_meta.json", "w") as f:
    json.dump({"doc_metadatas": doc_metadatas, "docs": docs}, f)

print("Embedding and indexing completed successfully on GPU!")
