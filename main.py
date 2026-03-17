from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


model = OllamaLLM(model="llama3.2")

template="You are an rules expert that answers questions about the rules of ultimate frisbee. " \
"Here are the rules: {rules}. Answer the following question: {question}" 


prompt = ChatPromptTemplate.from_template(template)

chain= prompt | model

result=chain.invoke({"rules":[], "question":"What is the maximum number of players on the field for each team?"})

print(result)