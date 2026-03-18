from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
import os
import re

load_dotenv()

rules_path = "data/usauRules.md"
if not os.path.exists(rules_path):
    rules_path = "data/usauRules.md"

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
        return []

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


def extract_rule_chunks(section_docs):
    chunks = []
    for section_doc in section_docs:
        chunks.extend(split_section_into_rule_blocks(section_doc))
    return chunks


# Keep full section context before structural rule chunking.
for section in md_sections:
    section.page_content = (section.page_content or "").strip()

md_chunks = extract_rule_chunks(md_sections)

pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME", "usau-rules")
namespace = os.getenv("PINECONE_NAMESPACE", "usau-rules-v1")
pinecone_cloud = os.getenv("PINECONE_CLOUD", "aws")
pinecone_region = os.getenv("PINECONE_REGION", "us-east-1")
pinecone_host = os.getenv("PINECONE_HOST", "").strip()
embed_model = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
embedding_dimension = int(os.getenv("PINECONE_EMBED_DIM", "1024"))
rebuild_namespace = os.getenv("PINECONE_REBUILD", "false").strip().lower() == "true"

if not pinecone_api_key:
    raise ValueError("Missing PINECONE_API_KEY in environment variables.")

pc = Pinecone(api_key=pinecone_api_key)

if pinecone_host:
    index = pc.Index(host=pinecone_host)
else:
    existing_indexes = pc.list_indexes().names()
    if index_name not in existing_indexes:
        print(f"Creating new cloud index: {index_name}...")
        pc.create_index(
            name=index_name,
            dimension=embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=pinecone_cloud, region=pinecone_region),
        )
    else:
        index_description = pc.describe_index(index_name)
        existing_dimension = (
            index_description.dimension
            if hasattr(index_description, "dimension")
            else index_description.get("dimension")
        )
        if existing_dimension and int(existing_dimension) != embedding_dimension:
            raise ValueError(
                f"Pinecone index '{index_name}' dimension is {existing_dimension}, "
                f"but model '{embed_model}' uses {embedding_dimension}. "
                "Set PINECONE_INDEX_NAME to a new index or set PINECONE_EMBED_DIM to match your index."
            )

    index = pc.Index(index_name)
stats = index.describe_index_stats()
namespace_count = stats.get("namespaces", {}).get(namespace, {}).get("vector_count", 0)
if rebuild_namespace and namespace_count > 0:
    print(f"Rebuilding namespace '{namespace}'...")
    index.delete(delete_all=True, namespace=namespace)
    namespace_count = 0

add_documents = namespace_count == 0

embeddings = OllamaEmbeddings(model=embed_model)

docs = []
ids = []
if add_documents:
    for i, section in enumerate(md_chunks):
        matched_rule_ids = [m.group("rule_id") for m in rule_pattern.finditer(section.page_content)]
        rule_id_summary = ", ".join(matched_rule_ids[:6])
        if len(matched_rule_ids) > 6:
            rule_id_summary += ", ..."

        document = Document(
            page_content=section.page_content,
            metadata={
                "source": "USAU",
                "section_title": section.metadata.get("section_title", ""),
                "rule_id": rule_id_summary,
                "chunk_id": str(i),
            },
            id=str(i)
        )
        ids.append(str(i))
        docs.append(document)

if pinecone_host:
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace,
        pinecone_api_key=pinecone_api_key,
        host=pinecone_host,
    )
else:
    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=namespace,
    )

if add_documents:
    print("Uploading USAU rules to Pinecone Cloud...")
    upsert_batch_size = int(os.getenv("PINECONE_UPSERT_BATCH", "16"))

    vectors = []
    for doc in docs:
        vectors.append({
            "id": doc.id,
            "values": embeddings.embed_query(doc.page_content),
            "metadata": {**dict(doc.metadata), "text": doc.page_content},
        })

    for start in range(0, len(vectors), upsert_batch_size):
        batch = vectors[start:start + upsert_batch_size]
        index.upsert(vectors=batch, namespace=namespace)

    print("Upload complete!")
    
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 24}
)


def cloud_similarity_search(query: str, k: int = 3):
    """Semantic search directly against the Pinecone cloud index."""
    return vector_store.similarity_search(query, k=k)
