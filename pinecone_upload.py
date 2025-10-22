import json
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer  # <-- CHANGED
from pinecone import Pinecone, ServerlessSpec
import config

# -----------------------------
# Config
# -----------------------------
DATA_FILE = "vietnam_travel_dataset.json"
BATCH_SIZE = 32 # This is fine, local processing is fast

# --- IMPORTANT ---
# This script assumes PINECONE_VECTOR_DIM = 384 in your config.py
# --- IMPORTANT ---

INDEX_NAME = config.PINECONE_INDEX_NAME
VECTOR_DIM = config.PINECONE_VECTOR_DIM  # Should be 384

# -----------------------------
# Initialize clients
# -----------------------------
pc = Pinecone(api_key=config.PINECONE_API_KEY)

# <-- CHANGED: Load the local Sentence Transformer model
# This will download the model (a few hundred MB) on its first run
print("Loading local embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2') 
print("Model loaded.")

# -----------------------------
# Create managed index if it doesn't exist
# -----------------------------

# --- CRITICAL CHECK ---
if VECTOR_DIM != 384:
    print(f"ERROR: Your VECTOR_DIM is {VECTOR_DIM}, but 'all-MiniLM-L6-v2' model requires 384.")
    print("Please change PINECONE_VECTOR_DIM = 384 in your config.py file.")
    exit()
# --- CRITICAL CHECK ---

existing_indexes = pc.list_indexes().names()
if INDEX_NAME not in existing_indexes:
    print(f"Creating managed index: {INDEX_NAME} with dimension {VECTOR_DIM}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM, # 384
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print("Index creation initiated. It may take a few minutes to be ready.")
    time.sleep(60) # Give Pinecone a moment to initialize
else:
    print(f"Index {INDEX_NAME} already exists.")

# Connect to the index
index = pc.Index(INDEX_NAME)

# -----------------------------
# Helper functions
# -----------------------------

# <-- CHANGED: Replaced Google function with local one
def get_embeddings(texts):
    """Generate embeddings using the local Sentence Transformer model."""
    try:
        # model.encode can take a list of texts directly
        # .tolist() converts the NumPy array to a plain list
        return model.encode(texts).tolist()
    except Exception as e:
        print(f"Error creating local embeddings: {e}")
        return None

def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

# -----------------------------
# Main upload
# -----------------------------
def main():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    items = []
    for node in nodes:
        semantic_text = node.get("semantic_text") or (node.get("description") or "")[:1000]
        if not semantic_text.strip():
            continue
        meta = {
            "id": node.get("id"),
            "type": node.get("type"),
            "name": node.get("name"),
            "city": node.get("city", node.get("region", "")),
            "tags": node.get("tags", [])
        }
        items.append((node["id"], semantic_text.strip(), meta))

    print(f"Preparing to upsert {len(items)} items to Pinecone using a local model...")
    
    for batch in tqdm(list(chunked(items, BATCH_SIZE)), desc="Uploading batches"):
        ids = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        metas = [item[2] for item in batch]
        
        embeddings = get_embeddings(texts)
        
        if embeddings is None:
            print(f"Failed to create embeddings for batch starting with ID: {ids[0]}. Skipping.")
            continue

        vectors = [
            {"id": _id, "values": emb, "metadata": meta}
            for _id, emb, meta in zip(ids, embeddings, metas)
        ]

        try:
            index.upsert(vectors)
        except Exception as e:
            print(f"Error upserting to Pinecone: {e}")
            print(f"Skipping batch starting with ID: {ids[0]}")
            
        # No more rate limit! We can remove the time.sleep()

    print("All items uploaded successfully.")

# -----------------------------
if __name__ == "__main__":
    main()