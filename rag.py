import os
import chromadb
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

def ask(query: str, db_dir: str = "chroma_db") -> tuple[str, list]:
    """
    Retrieves relevant chunks from the database and uses Mistral AI to answer the query.
    Returns a tuple of (answer, retrieved_documents).
    """
    load_dotenv()
    
    # 1. If database does not exist or is empty, return default message
    if not os.path.exists(db_dir) or not os.listdir(db_dir):
        return ("I could not find the answer in the uploaded document.", [])
        
    # 2. Initialize embedding model
    embedding_model = MistralAIEmbeddings(model="mistral-embed")
    
    # 3. Initialize Chroma and retrieve documents
    client = chromadb.PersistentClient(path=db_dir)
    docs = []
    try:
        vectorstore = Chroma(
            client=client,
            embedding_function=embedding_model
        )
        
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 10,
                "lambda_mult": 0.5
            }
        )
        docs = retriever.invoke(query)
    except Exception as e:
        # If any db query error occurs, return empty/fallback response
        return (f"Error accessing the vector database: {str(e)}", [])
    finally:
        client.close()
        
    # 4. If no documents retrieved, return default message
    if not docs:
        return ("I could not find the answer in the uploaded document.", [])
        
    # 5. Setup LLM and Prompt
    # Using mistral-small-latest as a stable and powerful model
    llm = ChatMistralAI(model="mistral-small-latest")
    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant for answering questions about the uploaded document.\n\n"
                "Use ONLY the provided context below to answer the user's question.\n"
                "Do not use any outside knowledge. Answer the question as directly and accurately as possible based on the context.\n\n"
                "CRITICAL: If the answer to the question cannot be found in the provided context, or if the context is empty, "
                "respond with EXACTLY: \"I could not find the answer in the uploaded document.\" and absolutely nothing else. "
                "Do not attempt to explain why, do not say you are sorry, and do not provide any extra text."
            ),
            (
                "human",
                "Context:\n{context}\n\n"
                "Question: {question}"
            )
        ]
    )
    
    # Combine content of retrieved documents
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Invoke model
    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })
    
    try:
        response = llm.invoke(final_prompt)
        answer = response.content.strip()
    except Exception as e:
        return (f"Error generating response from LLM: {str(e)}", docs)
    
    # Post-process fallback check
    # Ensure that if the LLM suggests it cannot find the answer, we return the exact requested string
    lowered_answer = answer.lower()
    if (
        "could not find" in lowered_answer or 
        "cannot find" in lowered_answer or
        "not found in the context" in lowered_answer or 
        "not present in the context" in lowered_answer or
        "context does not provide" in lowered_answer or
        "not mentioned" in lowered_answer or
        answer == ""
    ):
        answer = "I could not find the answer in the uploaded document."
        
    return answer, docs

    