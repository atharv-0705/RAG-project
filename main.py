from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

loader = PyPDFLoader("document loaders/SQL.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overla.pyp=50
)
chunks = splitter.split_documents(docs)

ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant."), 
     ("human", "{input}")]
)

model = ChatMistralAI(model="mistral-small-2506")

prompt = ChatPromptTemplate.format_prompt(input=docs[1].page_content)

result = model.invoke(prompt)

print(result.content)