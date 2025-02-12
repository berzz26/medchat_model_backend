from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
import os
from dotenv import load_dotenv
from src.helper import download_hugging_face_embeddings
from src.prompt import *
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from pinecone import Pinecone
from langchain.prompts import PromptTemplate
import uuid

# Load environment variables
load_dotenv()

app = FastAPI()

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Specify your React app's origin
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Initialize Pinecone
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_API_ENV")
)

# Load embeddings & vector store
embedding = download_hugging_face_embeddings()
index_name = "mchatbot"

docsearch = LangchainPinecone.from_existing_index(
    index_name=index_name,
    embedding=embedding,
)
PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}
# Load Llama model
llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={'max_new_tokens': 512, 'temperature': 0.8}
)

# Create the retrieval-based QA chain
qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 1}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)

# Define request schema
class ChatRequest(BaseModel):
    msg: str

@app.get("/")
def root():
    return {"message": "FastAPI server is running!"}

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        print("User Input:", request.msg)
        
        result = qa({"query": request.msg})
        cleaned_response = result["result"].replace("\n", " ")
        
        response_data = {
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": cleaned_response
        }
        
        print("Sending Response:", response_data)
        return response_data
        
    except Exception as e:
        print("Error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn app:app --host 0.0.0.0 --port 8080 --reload
