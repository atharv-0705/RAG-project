from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)

loader = PyPDFLoader("document loaders/SQL.pdf")
docs = loader.load()

chunks = splitter.split_documents(docs)

for i in chunks:
    print(i.page_content)
    print()
    print()
    print()

print(docs[1].page_content)

print(chunks[0].page_content)  