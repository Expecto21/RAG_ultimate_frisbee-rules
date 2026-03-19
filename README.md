Overview
This version provides a graphical user interface (GUI) built with Streamlit. It transforms the backend RAG logic into a functional web application for end-users.

Key Components
Frontend: Streamlit

State Management: Session State for persistent chat history

Inference: Connected to either local (Ollama) or cloud (Groq/OpenAI) backends

Features
Chat Interface: A familiar messaging UI for interacting with the rulebook.

Slang Translator: Integrated dictionary that maps player terminology to official rulebook language.

Source Transparency: Expandable UI components that display the exact rule text and metadata retrieved for every answer.

Caching: Optimized resource loading to prevent redundant model initialization.
