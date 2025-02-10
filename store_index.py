#ONLY RUN THIS TO UPDATE THE VECTOR EMBEDDINGS OR OVERRIDE IT
#RUNNING IT WILL CONSUME TIME AS THE DATA WILL BE UPSERT TO PINECONE VECTORDB


from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')



extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embedding = download_hugging_face_embeddings()


#Initializing the Pinecone
from pinecone import Pinecone

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),  # Ensure this points to the correct key
    environment=PINECONE_API_ENV  # Specify the environment
)

index_name = "mchatbot"

# Now proceed with the vector store
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,  # Ensure you're passing the correct documents here
    embedding=embedding,
    index_name=index_name,
    pinecone_api_key=os.getenv("PINECONE_API_KEY")  # Pass API key explicitly if needed
)

