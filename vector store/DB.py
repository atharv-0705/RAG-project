from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document

docs = [
    Document(
        page_content="SQL JOIN is used to combine rows from two or more tables based on a related column between them. Common types include INNER JOIN, LEFT JOIN, RIGHT JOIN, and FULL JOIN.",
        metadata={"source": "SQL.pdf"}
    ),

    Document(
        page_content="A SQL Subquery is a query nested inside another SQL query. It is commonly used in the WHERE or FROM clause to filter or process data based on the result of another query.",
        metadata={"source": "SQL.pdf"}
    ),

    Document(
        page_content="A SQL View is a virtual table created from the result of a SELECT statement. Views always display the latest data because they are generated dynamically whenever queried.",
        metadata={"source": "SQL.pdf"}
    ),
]

embedding_model = MistralAIEmbeddings(model="mistral-embed")

vectorstore = Chroma.from_documents(
    documents=docs, 
    embedding=embedding_model,
     persist_directory="chroma-db"
)

result = vectorstore.similarity_search("what is used for SQL JOIN?",k=2) # VECTOR STORE IS RESPONSIBLE FOR Retrival Augmented Generation (RAG) and it is used to store the embeddings of the documents and perform similarity search on them.

for r in result:
    print(r.page_content)
    print(r.metadata)
    
retriver = vectorstore.as_retriever()

docs = retriver.invoke("Explain difference between Union and Join.") # retriver.invoke() is used to retrieve the documents from the vector store based on the query provided.

for d in docs:
    print(d.page_content) 
