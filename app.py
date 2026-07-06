import os
import streamlit as st

from rag import ask
from create_database import create_database

st.set_page_config(

    page_title="RAG Assistant",
    page_icon="📚",
    layout="wide"

)

st.title("📚 RAG Document Assistant")

st.write("Upload a PDF and ask questions about it.")

#####################################################

if "messages" not in st.session_state:

    st.session_state.messages = []

#####################################################

with st.sidebar:

    st.header("Upload Document")

    uploaded_file = st.file_uploader(

        "Choose PDF",

        type=["pdf"]

    )

    if uploaded_file:

        os.makedirs("uploads", exist_ok=True)

        file_path = os.path.join(

            "uploads",

            uploaded_file.name

        )

        with open(file_path,"wb") as f:

            f.write(uploaded_file.getbuffer())

        if st.button("Create Vector Database"):

            with st.spinner("Processing PDF..."):

                chunks = create_database(file_path)

            st.success(f"{chunks} chunks indexed successfully.")

#####################################################

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.write(msg["content"])

#####################################################

query = st.chat_input("Ask anything about your document...")

if query:

    st.session_state.messages.append(

        {

            "role":"user",

            "content":query

        }

    )

    with st.chat_message("user"):

        st.write(query)

    with st.spinner("Thinking..."):

        answer, docs = ask(query)

    with st.chat_message("assistant"):

        st.write(answer)

        with st.expander("Retrieved Chunks"):

            for i, doc in enumerate(docs):

                st.markdown(f"### Chunk {i+1}")

                st.write(doc.page_content)

                st.caption(doc.metadata)

    st.session_state.messages.append(

        {

            "role":"assistant",

            "content":answer

        }

    )