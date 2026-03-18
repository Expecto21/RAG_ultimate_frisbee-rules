from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import os
import re

with open("Data/UsauRules.md", "r", encoding="utf-8") as f:
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

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100,
    separators=["\n\n", "\n", " "],
)
md_chunks = text_splitter.split_documents(rule_blocks)
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
    search_kwargs={"k": 8, "fetch_k": 24}
)
