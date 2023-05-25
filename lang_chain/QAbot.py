import os
import time
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma


# os.environ["OPENAI_API_KEY"] = ""

retriever = None

def initialize_retriever():
    embedding = OpenAIEmbeddings()
    global retriever
    vectordb = Chroma(persist_directory='db', embedding_function=embedding)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})


def generate_response(query):
    question_template = """請謹慎評估詢問的問題與給予之資料的相關性，並且只根據本段輸入文字訊息的內容進行回答，如果詢問的問題與提供的資料無關，請回答"您問的問題不在提供的資料當中"，另外也不要回答無關的答案：
    資料: {context}
    問題: {question}
    Answer:"""
    PROMPT = PromptTemplate(
        template=question_template, input_variables=["context", "question"]
    )
    docs = retriever.get_relevant_documents(query)
    
    chain = load_qa_chain(ChatOpenAI(temperature=0), chain_type="stuff", prompt=PROMPT)
    result = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
    return result['output_text']

def main():
    # Initialize the retriever
    initialize_retriever()

    # Continue the prompt until "exit" is entered
    while True:
        query = input("Ask me question (or 'end' to end): ")
        if query.lower() == 'end':
            print('Ending the program...')
            break
        
        start_time = time.time()
        response = generate_response(query)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Your question is : {query}\nAnswer: {response}\n Cost time = {elapsed_time}\n")

if __name__ == "__main__":
    main()
