from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document

docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"}),
]

embedding_model = MistralAIEmbeddings(model="mistral-embed")

vectorstore = Chroma.from_documents(
    documents=docs, 
    embedding=embedding_model,
     persist_directory="chroma-db"
)

result = vectorstore.similarity_search("what is used for data analysis?",k=2) # VECTOR STORE IS RESPONSIBLE FOR Retrival Augmented Generation (RAG) and it is used to store the embeddings of the documents and perform similarity search on them.

for r in result:
    print(r.page_content)
    print(r.metadata)
    
retriver = vectorstore.as_retriever()

docs = retriver.invoke("Explain neural networks.") # retriver.invoke() is used to retrieve the documents from the vector store based on the query provided.

for d in docs:
    print(d.page_content) 
