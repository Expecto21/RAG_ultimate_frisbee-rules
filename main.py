from vector import retriever
from google import genai
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

st.set_page_config(page_title="USAU Rules Bot", page_icon="🥏",layout="centered")


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

@st.cache_resource
def get_client():
    return genai.Client()

client = get_client()

def build_prompt(rules_context: str, question: str)-> str:
    return f"""You are an authoritative USA Ultimate rules official assisting players with in-game rule disputes and situational confusion.

Your primary goal is to resolve the user's specific scenario quickly, explain how the rules apply in plain English, and provide the exact official rule citations as proof.

CRITICAL INSTRUCTIONS:
1. Grounding: Rely STRICTLY on the provided Rules Context. Do not invent rules or borrow terminology from other sports. If the situation cannot be resolved from the context, state: "I cannot answer this scenario based on the provided rules."
2. Output Structure: Use the following headings for clarity:
   - **Ruling & Application**: 2–3 sentences explaining how the rule applies directly to this scenario in plain English. State clearly what call is made, who gets the disc, and how the stall restarts (distinguish contested vs. uncontested if applicable).
   - **Official Rule Citations**: Quote the exact operative clause(s) verbatim, prefixed with the official rule number (e.g., Rule 17.I.4.b).
3. Tone: Decisive, concise, and neutral. No conversational filler or introductory greetings.

Slang Glossary:
{slang_glossary}

Rules Context:
{rules_context}

Question: {question}
""" 






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

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("USAU Rules RAG")
st.markdown("Expert of the official USA Ultimate Rulebook.")

# CHANGE: Display previous messages from the history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# CHANGE: Replaced 'input()' loop with 'st.chat_input'
if question := st.chat_input("Ask about a rule (e.g., 'What happens on a strip?')"):
    
    # Display user question
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # CHANGE: Replaced print() statements with a UI Spinner and Chat Message
    with st.chat_message("assistant"):
        with st.spinner("Searching Rulebook..."):
            # 1. Retrieval
            rules = retriever.invoke(question)
            rules_context = format_rules_context(rules)
            
            # 2. Generation
            prompt_text = build_prompt(rules_context, question)
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt_text,
            )
            result = response.text
            
            # 3. Show Answer
            st.markdown(result)
            
            # CHANGE: Added an Expandable section to show the "Source Chunks" 
            # This is a huge "plus" for a resume to show transparency in RAG.
            with st.expander("View Referenced Rule Chunks"):
                for i, chunk in enumerate(rules, start=1):
                    st.write(f"**{i}. Rule {chunk.metadata.get('rule_id')}** ({chunk.metadata.get('section_title')})")
                    st.info(chunk.page_content)

    # Save assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": result})

