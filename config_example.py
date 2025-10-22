# config_example.py — copy to config.py and fill with real values.
NEO4J_URI = "neo4..."
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = ""

PINECONE_API_KEY = "pcsk_..." # your Pinecone API key
PINECONE_ENV = "us-east1-aws"   # example
PINECONE_INDEX_NAME = "vietnam-travel"
# PINECONE_VECTOR_DIM = 768       # adjust to embedding model used (text-embedding-3-large ~ 3072? check your model); we assume 1536 for common OpenAI models — change if needed.
PINECONE_VECTOR_DIM = 384      # adjust to embedding model used (all-MiniLM-L6-v2 requires 384); we assume 1536 for common OpenAI models — change if needed. 