import os
import bs4
from fastapi import FastAPI, HTTPException,Request
from fastapi.responses import HTMLResponse,FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Create FastAPI app
app = FastAPI()

# Serve static files from the frontend directory
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/")
async def home(request: Request):
    return FileResponse("frontend/base.html")

# Allow CORS for all origins (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model for input request
class QueryRequest(BaseModel):
    question: str

# Predefined questions and their responses
predefined_responses = {
    "hi": "Hello! How can I assist you today?",
    "hello": "Hello! How can I assist you today?",
    "hey": "Hi there! How can I help you?",
    "bye": "Goodbye! Have a great day!",
    "goodbye": "Goodbye! Take care!",
    "help": "I'm here to help! You can ask me about our company.",
    "thanks": "You're welcome! If you have more questions, feel free to ask.",
    "thank you": "You're welcome! I'm here to assist you.",
    "how are you?": "I'm just a program, but thanks for asking! How can I assist you?",
    "what's up?": "Not much! How can I help you today?",
    "what can you do?": "I can assist you with questions about our company and services.",
    "tell me about yourself": "I'm a virtual assistant here to help with your queries.",
    "can you help me?": "Of course! Please ask your question.",
    "hi there": "Hello! How can I help you today?",
    "good morning": "Good morning! How can I assist you?",
    "good evening": "Good evening! How can I help you?",
    "welcome": "Welcome! How can I assist you today?"
}


import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def crawl_website(base_url):
    visited = set()
    to_visit = [base_url]
    website_content = []

    while to_visit:
        url = to_visit.pop(0)
        if url in visited:
            continue

        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract text content from the page
            page_text = soup.get_text(separator="\n")
            website_content.append(page_text)

            # Find all links on the page and add them to the queue
            for link in soup.find_all("a", href=True):
                full_url = urljoin(base_url, link["href"])
                if base_url in full_url and full_url not in visited:
                    to_visit.append(full_url)

            visited.add(url)

        except Exception as e:
            print(f"Failed to retrieve or parse {url}: {e}")

    return "\n".join(website_content)  # Combine all pages' content



import os
from pathlib import Path
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def prepare_database():
    ABS_PATH = os.path.dirname(os.path.abspath(__file__))
    DB_DIR = os.path.join(ABS_PATH, "web_db")

    # Initialize the embedding function
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Check if the database directory already exists
    if Path(DB_DIR).exists():
        print("Database found, loading the existing vectorstore.")
        # Load the existing Chroma vectorstore with the embedding function
        vectorstore = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings  # Add the embedding function here
        )
        return vectorstore
    else:
        print("No existing database found, preparing a new vectorstore.")
        
        # Crawl the website and gather all content
        base_url = "https://www.conversedatasolutions.com"
        website_content = crawl_website(base_url)
        
        if not website_content:
            raise Exception("No content was retrieved from the website.")
        
        # Split the website content into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_text(website_content)
        
        if len(splits) == 0:
            raise Exception("No document chunks were created from the website content.")
        
        # Create a new Chroma vectorstore and persist it
        vectorstore = Chroma.from_texts(
            texts=splits,
            embedding=embeddings,
            persist_directory=DB_DIR
        )

        return vectorstore


# Prepare the vectorstore at startup
vectorstore = prepare_database()


@app.post("/ask")
async def ask_question(query: QueryRequest):
    prompt = query.question.lower()  # Convert question to lowercase to handle case insensitivity

    # Check if the question matches any predefined responses
    if prompt in predefined_responses:
        return {"response": predefined_responses[prompt]}
    
    retriever = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k": 3})

    # Get relevant documents based on the user's query
    relevant_docs = retriever.get_relevant_documents(prompt)

    if len(relevant_docs) == 0:
        raise HTTPException(status_code=404, detail="No relevant documents were retrieved from the vectorstore.")

    # Display retrieved documents for debugging
    context = "\n".join(doc.page_content for doc in relevant_docs)

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",temperature=0.2,max_output_tokens=500)

    system_prompt = (
    "You are a helpful assistant for Converse Data Solutions. Your responses depend on the user's query:\n"
    "1. If the query relates to website details, provide a concise answer based on the context.\n"
    "2. If the query is unclear or doesn't have meaning, respond with a concise request for clarification.\n"
    "3. If you don't know the answer, simply say you don't know.\n"
    "Use clear and concise language based on the following context:\n\n{context}\n\n"
)



    chat_prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )

    # Creating the chain
    question_answering_chain = create_stuff_documents_chain(llm, chat_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answering_chain)

    # Process the input query and display the result
    response = rag_chain.invoke({"input": prompt, "context": context})

    result = response.get("answer", "No answer found.")
    print(result)
    return {"response": result}

# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
