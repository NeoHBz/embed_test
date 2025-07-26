import numpy as np
import faiss
import os
import json

from sentence_transformers import SentenceTransformer

# Base cache path
base_dir = os.path.expanduser("~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots")

# Get the only snapshot folder (assumes one exists)
snapshots = os.listdir(base_dir)
assert len(snapshots) == 1, "Expected exactly one snapshot folder"
model_path = os.path.join(base_dir, snapshots[0])

# Load model
model = SentenceTransformer(model_path, local_files_only=True)

data_dir = "ttt"
index = faiss.read_index(data_dir + "_index.faiss")

# Load metadata
with open(data_dir + "_meta.json", "r") as f:
    metadata = json.load(f)
docs = metadata["docs"]
doc_metadatas = metadata["doc_metadatas"]

# Query and search
# Query and search
query = "maid lighted oil lamp"
query_vec = model.encode([query])
D, I = index.search(np.array(query_vec, dtype="float32"), k=10)

# Print top matches
print("Top matches:")
for idx, (i, distance) in enumerate(zip(I[0], D[0]), 1):
    print(f"{idx}. Doc: {doc_metadatas[i]['doc_id']}, Chunk: {doc_metadatas[i]['chunk_num']}, Distance: {distance:.4f}")
    print(f"   Text: {docs[i]}")
    print()

# Check if the exact phrase is found
print("Exact phrase search:")
for idx, (i, distance) in enumerate(zip(I[0], D[0]), 1):
    if "fighting were intensifying" in docs[i]:
        print(f"✓ Found exact phrase at rank {idx} with distance {distance:.4f}")
        print(f"  Text: {docs[i]}")
        break
else:
    print("✗ Exact phrase not found in top 10 results")

# Check if the exact sentence is in the results
target_sentence = "Then a detachment of imperial horsemen swept past her palanquin heading towards the rear of the column where the sounds of fighting were intensifying."
print("Checking for exact sentence:")
for idx, i in enumerate(I[0]):
    if docs[i] == target_sentence:
        print(f"Found exact sentence at rank {idx+1} with distance {D[0][idx]:.4f}")
        break
else:
    print("Exact sentence not found in top 10 results")
