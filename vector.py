from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import os
import re

load_dotenv()

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

pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME", "usau-rules")
namespace = os.getenv("PINECONE_NAMESPACE", "usau-rules-v1")
pinecone_cloud = os.getenv("PINECONE_CLOUD", "aws")
pinecone_region = os.getenv("PINECONE_REGION", "us-east-1")
embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

if not pinecone_api_key:
    raise ValueError("Missing PINECONE_API_KEY in environment variables.")
if not openai_api_key:
    raise ValueError("Missing OPENAI_API_KEY in environment variables.")

if embed_model == "text-embedding-3-small":
    embedding_dimension = 1536
elif embed_model == "text-embedding-3-large":
    embedding_dimension = 3072
else:
    raise ValueError(
        "Unsupported OPENAI_EMBED_MODEL. Use text-embedding-3-small or text-embedding-3-large."
    )

pc = Pinecone(api_key=pinecone_api_key)

existing_indexes = [idx["name"] for idx in pc.list_indexes().to_dict().get("indexes", [])]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=embedding_dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud=pinecone_cloud, region=pinecone_region),
    )

index = pc.Index(index_name)
stats = index.describe_index_stats()
namespace_count = stats.get("namespaces", {}).get(namespace, {}).get("vector_count", 0)
add_documents = namespace_count == 0

embeddings = OpenAIEmbeddings(model=embed_model, api_key=openai_api_key)

if add_documents:
    ids = []
    documents = []
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

vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    namespace=namespace,
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 24}
)
