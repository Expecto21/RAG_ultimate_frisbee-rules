from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever


model = OllamaLLM(model="llama3.2", temperature=0.2, repeat_penalty=1.15)

ULTIMATE_SLANG = {
    "greatest": "A player jumps from in-bounds, catches near the sideline, and releases a legal throw before landing out-of-bounds.",
    "calahan": "A defensive player catches the offense's pass in the offense's end zone for an immediate score.",
    "hospital pass": "A floaty or risky throw that exposes the receiver to heavy defensive pressure or contact.",
    "layout": "A fully extended dive attempt to catch or block a disc.",
    "sky": "To catch a disc over another player at the highest point.",
    "skyed": "A catch over another player that may involve vertical space/receiving foul considerations.",
    "hammer": "An overhand throw with the disc upside down in flight.",
    "strip": "A call about possession being dislodged; see possession and Rule 17.I.4.d context.",
    "universe": "Double game point.",
    "brick": "A pull that lands out of bounds or in the brick-mark area.",
    "ref": "Observer / Game Advisor context in a primarily self-officiated game.",
    "official": "Observer / Game Advisor context in a primarily self-officiated game.",
    "foul call": "Infraction / violation style player-initiated call.",
}

slang_glossary = "\n".join([f"- {term}: {definition}" for term, definition in ULTIMATE_SLANG.items()])

template="""Your job is to answer the user's question based STRICTLY on the provided rules context.

Constraints:
1. Use ONLY the provided rules context to answer the question. Do not assume intent or use outside knowledge of other sports.
2. If the answer is not in the context, say: "I cannot answer this based on the provided rules."
3. Response style ratio: about 80 percent synthesized explanation and application in plain English, and about 20 percent direct quoting.
4. Act like a translator: explain what the rules mean in practical terms for the user's specific scenario.
5. Always cite the specific rule numbers you used to form your answer.

Slang Glossary:
{slang_glossary}

Rules Context:
{rules_context}

Question: {question}
""" 


prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model


def format_rules_context(chunks):
    formatted_chunks = []
    for i, chunk in enumerate(chunks, start=1):
        section_title = chunk.metadata.get("section_title", "Unknown Section")
        rule_id = chunk.metadata.get("rule_id", "Unknown Rule")
        child_chunk_index = chunk.metadata.get("child_chunk_index", "N/A")
        chunk_id = chunk.metadata.get("chunk_id", "N/A")
        content = (chunk.page_content or "").strip()
        if not content:
            continue
        formatted_chunks.append(
            (
                f"[Chunk {i}] section={section_title} | rule={rule_id} "
                f"| child_chunk={child_chunk_index} | chunk_id={chunk_id}\n{content}"
            )
        )
    return "\n\n".join(formatted_chunks)

while True:
    print("\n-----------------------------------")
    question = input("Ask a question about the rules of ultimate frisbee (or type 'q' to quit): ")
    question = question.strip()
    if not question:
        continue

    if question.lower() == 'q':
        break

    rules = retriever.invoke(question)
    print("Referenced Chunks:")
    for i, chunk in enumerate(rules, start=1):
        section_title = chunk.metadata.get("section_title", "Unknown Section")
        rule_id = chunk.metadata.get("rule_id", "Unknown Rule")
        child_chunk_index = chunk.metadata.get("child_chunk_index", "N/A")
        chunk_id = chunk.metadata.get("chunk_id", "N/A")
        print(f"\n[{i}] section={section_title} | rule={rule_id} | child_chunk={child_chunk_index} | chunk_id={chunk_id}")
        print("Chunk Text:")
        print(chunk.page_content)

    print("\nAnswer:")
    rules_context = format_rules_context(rules)
    result=chain.invoke({"rules_context":rules_context, "question":question, "slang_glossary":slang_glossary})
    print(result)

