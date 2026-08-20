# RAG AI Assistant 🧠

An intelligent, autonomous AI assistant that allows users to upload PDF documents and ask questions based strictly on the uploaded content. Powered by Mistral AI, ChromaDB, and FastAPI, featuring a stunning glassmorphism user interface.

## Problem Statement
In today's information-heavy world, extracting precise answers from lengthy documents (like manuals, research papers, or legal contracts) is time-consuming. Traditional keyword search is often inadequate for complex queries, and generic AI chatbots hallucinate or lack context regarding private, specific documents.

## Objectives
- **Context-Aware Answers**: Provide an AI that answers questions based *only* on the provided document context.
- **Ease of Use**: Offer a seamless, drag-and-drop interface for users to upload documents.
- **Speed & Accuracy**: Utilize vector databases (ChromaDB) for fast semantic retrieval and advanced LLMs (Mistral) for accurate generation.

## Features
- 📄 **PDF Document Upload**: Drag and drop PDF files for immediate vectorization and indexing.
- 💬 **Interactive Chat Interface**: A modern, real-time chat UI with typing indicators and conversational flow.
- 🔍 **Source Transparency**: Every AI answer includes an expandable accordion showing the exact text chunks and page numbers retrieved from the source document.
- 🎨 **Glassmorphism UI**: A visually stunning frontend with an animated ambient orb background and frosted glass cards.
- ⚙️ **Session Management**: Easily reset the database and clear uploads with a single click.

## System Architecture
The application follows a client-server architecture:
- **Client**: A lightweight, single-page application built with React (via CDN) and Vanilla CSS, running entirely in the browser.
- **Server**: A robust FastAPI backend that handles file processing, database orchestration, and API communication.

## AI Pipeline Architecture
1. **Document Loading**: PyPDFLoader extracts text from the uploaded PDF.
2. **Chunking**: RecursiveCharacterTextSplitter breaks the text into overlapping 1000-character chunks.
3. **Embedding**: MistralAIEmbeddings (`mistral-embed`) converts chunks into dense vector representations.
4. **Vector Storage**: ChromaDB stores the vectors persistently on disk.
5. **Retrieval**: When a query is made, Chroma uses MMR (Maximal Marginal Relevance) to fetch the top 4 most relevant chunks.
6. **Generation**: A prompt restricts the `mistral-small-latest` LLM to answer using *only* the retrieved context.

## 🖥️ Frontend Overview
The frontend is a single `index.html` file served by FastAPI. It leverages **React** for component state management (handling chats, uploads, and toast notifications) without the complexity of a build step (Node.js/Webpack). The design utilizes a deep dark theme with vibrant violet and cyan accents (Glassmorphism).

## 🛠️ Tech Stack
- **Frontend**: HTML5, CSS3, JavaScript (React via CDN, Babel)
- **Backend Framework**: Python, FastAPI, Uvicorn
- **AI / LLM Orchestration**: LangChain, Mistral AI
- **Vector Database**: ChromaDB
- **Document Processing**: PyPDF

## 📁 Folder Structure
```text
📦 RAG Project
 ┣ 📂 static
 ┃ ┗ 📜 index.html           # The Glassmorphism React frontend
 ┣ 📂 uploads                # Temporary storage for uploaded PDFs
 ┣ 📂 chroma_db              # Persistent local vector database storage
 ┣ 📜 app.py                 # FastAPI backend server and routes
 ┣ 📜 create_database.py     # Logic for parsing PDFs and building the Chroma DB
 ┣ 📜 rag.py                 # Logic for embedding queries and prompting Mistral AI
 ┣ 📜 requirements.txt       # Python dependencies
 ┣ 📜 .env                   # Environment variables (Mistral API key)
 ┗ 📜 .gitignore             # Git ignore rules
```

## ⚙️ Backend Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd "RAG Project"
   ```

2. **Set up a virtual environment** (Recommended):
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate  # On Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**:
   Create a `.env` file in the root directory and add your Mistral API key:
   ```env
   MISTRAL_API_KEY=your_api_key_here
   ```

5. **Run the Application**:
   ```bash
   uvicorn app:app --port 2220 --reload
   ```
   The app will be available at `http://127.0.0.1:2220`.

## API Endpoints
- `GET /` : Serves the main web UI.
- `GET /health` : Returns system health and database status.
- `POST /upload` : Accepts a PDF file, parses it, and builds the Chroma vector database.
- `POST /ask` : Accepts a JSON payload `{"question": "..."}`, retrieves context, and returns the AI answer + sources.
- `DELETE /reset` : Clears the current vector database and deletes uploaded files.

## 🔄 Document Query Workflow
1. User uploads a PDF via the frontend UI.
2. The `/upload` endpoint saves the file, splits it into chunks, and stores vectors in ChromaDB.
3. User types a question in the chat.
4. The `/ask` endpoint embeds the question, queries ChromaDB for the closest semantic chunks, and passes the context to Mistral AI.
5. Mistral AI generates a response strictly based on the context, which is sent back to the frontend along with the source texts.

## Deployment
This app can be deployed on platforms like **Render**:
1. Connect your GitHub repository to Render as a "Web Service".
2. Set the build command to `pip install -r requirements.txt`.
3. Set the start command to `uvicorn app:app --host 0.0.0.0 --port $PORT`.
4. Add your `MISTRAL_API_KEY` in the Render Environment Variables tab.
*(Note: Render's free tier uses an ephemeral filesystem, meaning uploaded PDFs and the vector database will reset upon server restart unless a persistent disk is attached).*

---

## Author
**Atharv Gupta**
- **LinkedIn**: [Atharv Gupta | LinkedIn](https://www.linkedin.com/in/atharv-gupta-45a37b36a/)
- **Email**: atharvgupta0705@gmail.com
