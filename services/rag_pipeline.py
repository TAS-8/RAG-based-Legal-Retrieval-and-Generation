import numpy as np
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document # Import Document
from retrieval import retrieve_documents
import re


# Step 1: Setup LLM (Use Deepseek R1 with Groq)
from dotenv import load_dotenv
load_dotenv()

import os
llm_model = ChatGroq(model="deepseek-r1-distill-llama-70b")

# Step 2: Get context
def get_context(documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    return context

# Step 3: Answer Question
custom_prompt_template = """
You are an AI assistant designed to provide clear, accurate, and concise answers to a wide range of questions.

Guidelines:
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Don't provide anything out of the given context

Question: {question}
Context: {context}
Answer:
"""

def answer_query(query, model_name, k=10):
    # Use the retrieve_documents function from retrieval.py
    retrieved_results = retrieve_documents(query, model_name)
    print(model_name)

    # Format the results into Langchain Document objects
    langchain_docs = []
    for res in retrieved_results:
        langchain_docs.append(Document(page_content=res['text'], metadata={
            "id": res.get('id'),
            "part": res.get("part"),
            "chapter": res.get("chapter"),
            "section": res.get("section"),
            "title": res.get("title"),
            "source": res.get("source"),
            "distance": res.get("distance"),
            "score": res.get("score")
        }))


    context = get_context(langchain_docs)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | llm_model
    result = chain.invoke({"question": query, "context": context})
    full_response = result.content
    final_answer = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()

    return final_answer


query = "I want to learn about criminal procedure in Pakistan."
model_to_use = 'legalbert'  # Choose the model you want to use, e.g., 'glove', 'legalbert', etc.
answer = answer_query(query, model_to_use, k=10) # Use your desired k value

print("Query:", query)
print("Answer:", answer)