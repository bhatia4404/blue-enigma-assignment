# Blue Enigma - Hybrid AI Travel Assistant

This project implements a hybrid AI travel assistant that answers user queries about Vietnam travel. It utilizes a combination of semantic vector search (Pinecone) and knowledge graph context (Neo4j) to provide relevant information, which is then synthesized by a Large Language Model (LLM) into a coherent answer.

## Project Goal

The initial goal was to debug and complete a semi-functional system. The system originally relied on OpenAI APIs for embeddings and chat generation. However, due to API quota limitations (`insufficient_quota`), a major re-engineering effort was undertaken.

## Final Architecture & Key Improvements

The final version of this project achieves the original goal while making significant improvements:

1.  **100% Local & Free:** The entire pipeline now runs using **free, open-source models locally**.
    - **Embeddings:** Uses `sentence-transformers` (`all-MiniLM-L6-v2` model) for generating embeddings, eliminating the OpenAI embedding API dependency.
    - **Chat:** Uses `Ollama` to serve the `llama3` LLM locally, eliminating the OpenAI chat API dependency.
      This makes the assistant cost-free, private, and resilient to API quotas or internet issues (after initial model downloads).
2.  **Streaming Output:** Implemented streaming responses from Ollama, providing a much better user experience by showing the answer as it's generated, especially important when running on slower CPU hardware.
3.  **Embedding Caching:** Added caching for user query embeddings to avoid redundant computations and slightly improve performance.
4.  **Advanced Prompt Engineering:** Significantly refined the prompt sent to the local LLM (`llama3`), providing stricter context limitations, clearer instructions on using Pinecone vs. Neo4j data, and a structured task list (Chain-of-Thought) to improve the relevance, accuracy, and structure (especially for itineraries) of the generated answers.
5.  **Robustness:** Added better error handling for database connections (Pinecone, Neo4j) and the Ollama server connection.
6.  **Dependency Management:** Updated `requirements.txt` to reflect the final set of libraries used.

## Files

- `hybrid_chat.py`: The main interactive chat script. (Final Improved Version)
- `pinecone_upload.py`: Script to generate embeddings locally and upload data to Pinecone. (Final Improved Version)
- `load_to_neo4j.py`: Script to load the dataset into Neo4j. (Provided)
- `visualize_graph.py`: Script to generate an HTML visualization of the Neo4j graph. (Provided, Bug Fixed)
- `config.py`: Configuration file for API keys and database credentials (generated from `config.py.sample`).
- `vietnam_travel_dataset.json`: The source data file. (Provided)
- `requirements.txt`: List of required Python packages. (Final Updated Version)
- `improvements.md`: Detailed documentation of the fixes and improvements made.

---

## Setup and Running Instructions

### Local Setup (CPU - Potentially Slow LLM Responses)

Follow these steps to run the project on your local machine.

1.  **Clone/Download:** Get all the project files.
2.  **Create Environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Install Ollama:**
    - Download and install from [https://ollama.com/](https://ollama.com/).
    - **If you already have Ollama:** Simply ensure the Ollama application/server is running in the background. You can check this in your system tray or task manager. If it is not running, run `ollama serve` to start it.
5.  **Pull LLM Model:**
    ```bash
    ollama pull llama3
    ```
    _(The `all-MiniLM-L6-v2` embedding model will be downloaded automatically by the Python scripts on first run)._
6.  **Configure:**
    - Copy `config_example.py` to `config.py`.
    - Edit `config.py` and add your **Pinecone API Key** and **Neo4j connection details** (URI, User, Password).
    - Verify `PINECONE_VECTOR_DIM = 384` in `config.py`.
7.  **Populate Databases:**
    - Ensure your Neo4j server is running.
    - _(Optional but Recommended)_ If you previously ran this with different embedding dimensions, **delete** the `vietnam-travel` index from your Pinecone dashboard.
    ```bash
    python load_to_neo4j.py
    python pinecone_upload.py
    ```
8.  **Run Chat:**
    ```bash
    python hybrid_chat.py
    ```
    **Note:** LLM response generation might take **1-3 minutes per query** on a CPU. Use the Colab instructions for a faster experience.

---

### Recommended: Running on Google Colab (GPU - Fast LLM Responses)

This method uses Google Colab's free T4 GPU for significantly faster LLM responses (typically under 5 seconds).

**Steps:**

1.  **Open Colab:** Go to [https://colab.research.google.com/](https://colab.research.google.com/) -> "New notebook".
2.  **Set Runtime:** Menu -> **Runtime -> Change runtime type** -> Select **`T4 GPU`**.
3.  **Upload Files:** Use the **folder icon** (left sidebar) -> **"Upload"** button to upload:
    - `config.py` (Must contain your valid keys/credentials!)
    - `hybrid_chat.py`
    - `pinecone_upload.py`
    - `load_to_neo4j.py`
    - `vietnam_travel_dataset.json`
    - Click the **"Refresh"** button in the file explorer.
4.  **Run Setup Cell:** Paste and run the following in a single **Code Cell**:

    ```python
    # Create requirements.txt
    requirements_content = """
    neo4j
    pyvis==0.3.1
    networkx==3.1
    Network
    python-dotenv
    pinecone-client==4.1.0
    tqdm
    sentence-transformers==2.7.0
    ollama==0.2.1
    """
    with open("requirements.txt", "w") as f:
        f.write(requirements_content)

    # Install Python libraries
    !pip install -q -r requirements.txt
    print("✅ Python libraries installed.")

    # Install and start Ollama
    !curl -fsSL [https://ollama.com/install.sh](https://ollama.com/install.sh) | sh
    import asyncio
    async def run_ollama_serve():
        process = await asyncio.create_subprocess_shell(
            'ollama serve', stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        await asyncio.sleep(5) # Wait for server init
        print("✅ Ollama server started.")
    await run_ollama_serve()

    # Pull the Llama3 model
    !ollama pull llama3
    print("✅ Llama3 model pulled.")
    ```

5.  **Run Data Population Cell:** Paste and run in a new **Code Cell**:

    ```python
    # Ensure Neo4j server is accessible from Colab
    # (May require adjusting Neo4j config or using a cloud instance like AuraDB Free)

    # Delete existing Pinecone index 'vietnam-travel' via Pinecone console if it has wrong dimension (should be 384)

    print("--- Running Neo4j Data Loader ---")
    !python /content/load_to_neo4j.py
    print("\n✅ Neo4j data loading complete (check for errors).")

    print("\n--- Running Pinecone Data Loader ---")
    !python /content/pinecone_upload.py
    print("\n✅ Pinecone data loading complete (check for errors).")
    ```

6.  **Run Chat Script Cell:** Paste and run in a new **Code Cell**:
    ```python
    # Run the main chat script!
    !python /content/hybrid_chat.py
    ```
7.  Interact with the assistant in the cell output. Responses should be fast.
