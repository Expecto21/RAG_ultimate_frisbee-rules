Overview
This version implements a Retrieval-Augmented Generation (RAG) system using a local vector store. It is designed for privacy and offline use, running entirely on the user's hardware.

Key Components
LLM: Ollama (Llama 3.2)

Vector Store: ChromaDB (Persistent local storage)

Embeddings: Ollama (mxbai-embed-large)

Data Source: Structured Markdown converted from USAU PDF

Features
Local Processing: No data leaves the machine, ensuring privacy and zero API costs.

Regex Parsing: Custom Python logic to extract specific Rule IDs and maintain section hierarchy.

MMR Retrieval: Uses Maximum Marginal Relevance to provide diverse rule context to the model.
