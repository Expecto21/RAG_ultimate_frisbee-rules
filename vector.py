from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("data/USAU.csv")
embeddings=OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    ids=[]
    documents=[]
    for i, row in df.iterrows():
        document = Document(
            page_content=row.get("text", ""),
            metadata={
                "source": row.get("source", "USAU"),
                "rulebook_version": row.get("rulebook_version", ""),
                "section_number": row.get("section_number", ""),
                "section_title": row.get("section_title", ""),
                "rule_id": row.get("rule_id", ""),
                "page": row.get("page", ""),
                "chunk_id": row.get("chunk_id", i),
            },
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)

vector_store = Chroma(
    collection_name="USAU_rules",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)
