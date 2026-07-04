from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import CharacterTextSplitter

splitters = CharacterTextSplitter(
    separator = "",
    chunk_size=1000, 
    chunk_overlap=10
)

loader = TextLoader("Note.txt", encoding="utf-8")

docs = loader.load()
chunks = splitters.split_documents(docs)

for i in chunks:
    print(i.page_content)
    print()
    print()
    print()

print(docs[0].page_content) # docs = Metadata + Page_content

print(chunks)