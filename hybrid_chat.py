# hybrid_chat.py
import json
from typing import List
import ollama
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from neo4j import GraphDatabase
import config

# -----------------------------
# Config
# -----------------------------
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # Embedding model used in pinecone_upload.py
CHAT_MODEL = "llama3"                  # Local chat model pulled via Ollama
TOP_K = 5                              # Number of results to fetch from Pinecone
INDEX_NAME = config.PINECONE_INDEX_NAME

# -----------------------------
# Initialize clients & Cache
# -----------------------------
pc = Pinecone(api_key=config.PINECONE_API_KEY)

print(f"Loading local embedding model: {EMBED_MODEL_NAME}...")
try:
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
except Exception as e:
    print(f"Error loading SentenceTransformer model: {e}")
    print("Make sure 'sentence-transformers' is installed (`pip install sentence-transformers`)")
    exit()
print("Embedding model loaded.")

embedding_cache = {} # Improvement 1: Cache dictionary

# --- Pinecone Connection ---
try:
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Error: Index '{INDEX_NAME}' does not exist in Pinecone.")
        print("Please run 'pinecone_upload.py' successfully first.")
        exit()
    print(f"Connecting to Pinecone index: {INDEX_NAME}...")
    index = pc.Index(INDEX_NAME)
    print("Connected to Pinecone.")
except Exception as e:
    print(f"Error connecting to Pinecone: {e}")
    print("Check your PINECONE_API_KEY and INDEX_NAME in config.py")
    exit()

# --- Neo4j Connection ---
try:
    driver = GraphDatabase.driver(
        config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )
    driver.verify_connectivity() # Check if connection is valid
    print("Connected to Neo4j.")
except Exception as e:
    print(f"Error connecting to Neo4j: {e}")
    print("Check your NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD in config.py")
    print("Ensure the Neo4j database server is running.")
    exit()

# -----------------------------
# Helper functions
# -----------------------------

def embed_text(text: str) -> List[float]:
    """Get embedding for a text string using the local model with cache."""
    # Improvement 1: Check cache first
    if text in embedding_cache:
        return embedding_cache[text]
    else:
        try:
            embedding = embed_model.encode(text).tolist()
            embedding_cache[text] = embedding # Store in cache before returning
            return embedding
        except Exception as e:
            print(f"Error generating embedding for text: '{text[:50]}...' - {e}")
            return None # Return None on failure

def pinecone_query(query_text: str, top_k=TOP_K):
    """Query Pinecone index using embedding."""
    vec = embed_text(query_text) # Uses cached embedding if available
    if vec is None:
        return [] # Return empty list if embedding failed

    try:
        res = index.query(
            vector=vec,
            top_k=top_k,
            include_metadata=True,
            include_values=False
        )
        return res.get("matches", []) # Safely get matches, default to empty list
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return []

def fetch_graph_context(node_ids: List[str]):
    """Fetch neighboring nodes and source node names from Neo4j."""
    facts = []
    node_names = {} # To store names for better prompt context

    if not node_ids: # Don't query Neo4j if Pinecone returned nothing
        return facts

    try:
        with driver.session() as session:
            # First, get names of the source nodes found by Pinecone
            name_query = "MATCH (n:Entity) WHERE n.id IN $node_ids RETURN n.id AS id, n.name AS name"
            name_results = session.run(name_query, node_ids=node_ids)
            for record in name_results:
                node_names[record["id"]] = record["name"]

            # Then, get relationships for each source node
            for nid in node_ids:
                rel_query = (
                    "MATCH (n:Entity {id:$nid})-[r]-(m:Entity) "
                    "RETURN type(r) AS rel, labels(m) AS labels, m.id AS id, "
                    "m.name AS name, m.type AS type, m.description AS description "
                    "LIMIT 5" # Limit relations per node to keep context concise
                )
                recs = session.run(rel_query, nid=nid)
                for r in recs:
                    facts.append({
                        "source": nid,
                        "source_name": node_names.get(nid, nid), # Use fetched name
                        "rel": r["rel"],
                        "target_id": r["id"],
                        "target_name": r["name"],
                        "target_desc": (r.get("description", "") or "")[:200], # Shorter description
                        "labels": r.get("labels", [])
                    })
        return facts
    except Exception as e:
        print(f"Error querying Neo4j: {e}")
        return [] # Return empty list on failure


def build_prompt(user_query, pinecone_matches, graph_facts):
    """Build a more detailed chat prompt specifically tuned for itinerary generation."""

    # --- ENHANCED System Prompt for Itinerary Planning ---
    system = (
        "You are a specialized travel assistant for Vietnam. Your knowledge is **strictly limited** to the information provided below in the 'semantic matches' and 'graph facts' sections. "
        "Your primary goal is to answer the user's query accurately based **only** on this context. "
        "\n\n**Context Usage Rules:**\n"
        "* Use semantic matches (Pinecone) to identify the core locations, themes (e.g., romantic, adventure), and concepts relevant to the user's query.\n"
        "* Use graph facts (Neo4j) to find related details: activities available in a city, attractions located near hotels, connections between places.\n"
        "* Cite node IDs (like 'city_hanoi' or 'attraction_1') **every time** you mention a specific place, hotel, or activity found in the context.\n"
        "\n\n**Itinerary Generation Rules (If applicable):**\n"
        "* If the user asks for an itinerary, structure the response clearly day-by-day (e.g., **Day 1: Arrival in [City Name] (city_id)**).\n"
        "* For each day, suggest **1-2 specific activities, attractions, or notable hotels** *directly mentioned in the provided context*, citing their node IDs.\n"
        "* Try to use graph facts to make logical suggestions (e.g., 'Visit attraction_X which is Located_In city_Y').\n"
        "* Ensure the suggestions align with the theme of the user's request (e.g., if they ask for 'romantic', prioritize items tagged as romantic or suggest activities suitable for couples).\n"
        "* Be concise but provide enough detail for the user to understand the suggestion.\n"
        "\n\n**General Rules:**\n"
        "* If the provided context does not contain enough information to answer fully or create a meaningful itinerary, **explicitly state that** (e.g., 'Based on the provided context, I don't have enough information about X...').\n"
        "* **Do not** use any prior knowledge or information outside the provided context. **Do not** make up places, details, or relationships."
    )

    vec_context = []
    if pinecone_matches:
        for m in pinecone_matches:
            meta = m.get("metadata", {})
            name = meta.get('name', 'N/A')
            type = meta.get('type', 'N/A')
            city = meta.get('city', '')
            # Include tags if they exist, as they might be relevant for themes
            tags = meta.get('tags', [])
            snippet = f"- ID: {m['id']}, Name: {name}, Type: {type}"
            if city:
                snippet += f", City: {city}"
            if tags:
                snippet += f", Tags: {tags}"
            vec_context.append(snippet)
    else:
        vec_context.append("- No relevant semantic matches found.")


    graph_context = []
    if graph_facts:
        for f in graph_facts:
            source_name = f.get('source_name', f.get('source', 'N/A'))
            rel = f.get('rel', 'RELATED_TO')
            target_id = f.get('target_id', 'N/A')
            target_name = f.get('target_name', 'N/A')
            target_desc = f.get('target_desc', '')
            graph_context.append(
                f"- {f.get('source', 'N/A')} ({source_name}) --[{rel}]--> {target_id} ({target_name}): {target_desc}"
            )
    else:
        graph_context.append("- No relevant graph facts found.")

    # --- ENHANCED User Task for Itinerary ---
    prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content":
         f"User query: {user_query}\n\n"
         "### Context from Semantic Search (Relevant Places/Concepts/Themes):\n" + "\n".join(vec_context) + "\n\n"
         "### Context from Knowledge Graph (Connections and Details):\n" + "\n".join(graph_context) + "\n\n"
         "### Your Task:\n"
         "1.  Analyze the user query, paying close attention to requested themes (e.g., romantic, adventure, budget) and duration.\n"
         "2.  Identify the most relevant entities (cities, attractions, hotels) from the Semantic Search context that match the query.\n"
         "3.  Use the Knowledge Graph context to find specific activities, nearby points of interest, or connections related to those entities.\n"
         "4.  Synthesize this information to **directly answer the user query**. \n"
         "5.  **If the query asks for an itinerary:**\n"
         "    a. Create a **day-by-day plan** based *only* on the provided context.\n"
         "    b. For each day, suggest **1-2 specific, relevant activities/attractions/hotels** from the context, **citing their node IDs**.\n"
         "    c. Use graph facts to make logical connections (e.g., suggest an activity available in the chosen city).\n"
         "    d. Ensure the plan fits the duration and theme requested.\n"
         "6.  Cite node IDs for **all** specific entities mentioned in your response.\n"
         "7.  If the context is insufficient to create a reasonable plan or answer the query, clearly state the limitations.\n"
         "\n**Answer:**" # Signal the model to start generating the final answer here
         }
    ]
    return prompt

# Improvement 2: Streaming chat function
def stream_chat(prompt_messages):
    """Call local Ollama model and stream the response."""
    full_response = ""
    try:
        # Check if Ollama server is running before making the call
        # This sends a lightweight request to see if the server responds.
        ollama.list() 
        
        stream = ollama.chat(
            model=CHAT_MODEL,
            messages=prompt_messages,
            stream=True,
            options={"temperature": 0.2}
        )

        print("\n=== Assistant Answer ===\n")
        for chunk in stream:
            part = chunk.get('message', {}).get('content', '')
            if part:
                print(part, end='', flush=True) # Print each part immediately
                full_response += part
        print() # Add a newline at the end

    except ConnectionRefusedError: # Specific error if server isn't running
         print(f"\n--- OLLAMA CONNECTION ERROR ---")
         print("Connection refused. Is the Ollama application running?")
         print("--------------------------------\n")
         return "Error: Could not connect to local chat model. Please ensure Ollama is running."
    except Exception as e: # Catch other potential errors
        print(f"\n--- OLLAMA ERROR ---")
        print(f"Error calling Ollama: {e}")
        # Consider checking if the specific model exists if it's a model not found error
        if "model" in str(e) and "not found" in str(e):
             print(f"Model '{CHAT_MODEL}' not found. Did you pull it? (e.g., 'ollama pull {CHAT_MODEL}')")
        print("---------------------\n")
        # Return partial response if any was collected before error
        if full_response:
             print("\n(Response may be incomplete due to error)")
             return full_response 
        return f"Error during Ollama chat generation: {e}"

    return full_response

# -----------------------------
# Interactive chat loop
# -----------------------------
def interactive_chat():
    """Main loop to handle user queries."""
    print("\n-------------------------------------------")
    print(" Hybrid Vietnam Travel Assistant (LOCAL)")
    print("-------------------------------------------")
    print("Enter your travel question below. Type 'exit' or 'quit' to end.")

    while True:
        try:
            query = input("\n➡️ Enter your travel question: ").strip()
            if not query:
                continue
            if query.lower() in ("exit","quit"):
                print("👋 Exiting assistant. Goodbye!")
                break

            print("\n🔍 Searching for relevant information...")
            matches = pinecone_query(query, top_k=TOP_K)
            match_ids = [m["id"] for m in matches if "id" in m] # Ensure 'id' exists

            print("🧠 Fetching contextual details...")
            graph_facts = fetch_graph_context(match_ids)

            print("📝 Building prompt for the assistant...")
            prompt = build_prompt(query, matches, graph_facts)

            # print("\n--- DEBUG: Full Prompt ---")
            # print(json.dumps(prompt, indent=2))
            # print("-------------------------\n")

            print("💬 Generating response (this might take a moment on CPU)...")
            answer = stream_chat(prompt) # Call the streaming function

            # Printing is handled within stream_chat
            print("\n--- End of Response ---")

        except KeyboardInterrupt: # Allow graceful exit with Ctrl+C
             print("\n👋 Exiting assistant. Goodbye!")
             break
        except Exception as e: # Catch any unexpected errors in the loop
             print(f"\n🚨 An unexpected error occurred: {e}")
             print("Please try again.")

# -----------------------------
# Main execution block
# -----------------------------
if __name__ == "__main__":
    try:
        interactive_chat()
    finally:
        # Ensure Neo4j driver is closed properly on exit
        if 'driver' in globals() and driver:
            driver.close()
            print("\nNeo4j connection closed.")