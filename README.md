Overview
This version transitions the vector storage to the cloud using Pinecone. It is built for scalability and allows multiple clients to access the same centralized knowledge base.

Key Components
LLM: Groq (Llama 3.1/3.3) or OpenAI

Vector Store: Pinecone (Serverless index)

Embeddings: OpenAI (text-embedding-3-small) or Voyage AI

Orchestration: LangChain

Features
Cloud Infrastructure: Decouples storage from the local machine, allowing for faster updates to the ruleset.

Hybrid Search: Optimized for combining semantic vector matches with keyword-based lookups.

High Availability: Provides a persistent endpoint that can be accessed via API keys without local database management.
