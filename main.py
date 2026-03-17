from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever


model = OllamaLLM(model="llama3.2")

template="You are an rules expert that answers questions about the rules and situations of ultimate frisbee. If you don't know the answer, say you don't know. " \
"The user may ask about specific scenarios. Find the rules that apply to the situation and explain what should happen according to the rules." \
"Use the the rules to construct your answer. Figure out the meaning of the question and the relevant rules and not just the litteral words in the question." \
"Here are the rules: {rules}. Answer the following question: {question}" 


prompt = ChatPromptTemplate.from_template(template)

chain= prompt | model

while True:
    print("\n\n-----------------------------------")
    question = input("Ask a question about the rules of ultimate frisbee (or type 'q' to quit): ")
    print("\n\n")
    if question.lower() == 'q':
        break

    rules = retriever.invoke(question)
    result=chain.invoke({"rules":rules, "question":question})
    print(result)

