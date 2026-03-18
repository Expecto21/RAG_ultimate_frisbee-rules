from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
import os
import re
from pathlib import Path

rules_path = Path(__file__).resolve().parent / "data" / "usauRules.md"

with open(rules_path, "r", encoding="utf-8") as f:
    md_text = f.read()

headers_to_split_on = [("##", "section_title")]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_sections = splitter.split_text(md_text)

rule_pattern = re.compile(r"^(?P<rule_id>\d+\.[A-Z](?:\.\d+)*(?:\.[a-z])?)\.\s", re.MULTILINE)


def split_section_into_rule_blocks(section_doc):
    text = (section_doc.page_content or "").strip()
    if not text:
        return []

    matches = list(rule_pattern.finditer(text))
    if not matches:
        return [section_doc]

    blocks = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block_text = text[start:end].strip()
        if not block_text:
            continue

        metadata = dict(section_doc.metadata)
        metadata["rule_id"] = match.group("rule_id")
        blocks.append(Document(page_content=block_text, metadata=metadata))

    return blocks


rule_blocks = []
for section in md_sections:
    rule_blocks.extend(split_section_into_rule_blocks(section))

def split_rule_blocks_consistently(documents, chunk_size=700, chunk_overlap=100):
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []
    step = chunk_size - chunk_overlap

    for doc in documents:
        text = (doc.page_content or "").strip()
        if not text:
            continue

        if len(text) <= chunk_size:
            metadata = dict(doc.metadata)
            metadata["child_chunk_index"] = 0
            chunks.append(Document(page_content=text, metadata=metadata))
            continue

        chunk_index = 0
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                metadata = dict(doc.metadata)
                metadata["child_chunk_index"] = chunk_index
                chunks.append(Document(page_content=chunk_text, metadata=metadata))

            if end >= len(text):
                break

            start += step
            chunk_index += 1

    return chunks


md_chunks = split_rule_blocks_consistently(rule_blocks, chunk_size=700, chunk_overlap=100)
embeddings=OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    ids=[]
    documents=[]
    for i, section in enumerate(md_chunks):
        document = Document(
            page_content=section.page_content,
            metadata={
                "source": "USAU",
                "section_title": section.metadata.get("section_title", ""),
                "rule_id": section.metadata.get("rule_id", ""),
                "chunk_id": i,
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
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 24}
)
