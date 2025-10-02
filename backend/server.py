from fastapi import FastAPI, APIRouter, HTTPException, File, UploadFile, Form
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import asyncio
import PyPDF2
import io
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import chromadb
from chromadb.config import Settings
from openai import AsyncOpenAI

ROOT_DIR = Path(__file__).parent

# Load environment variables
if os.path.exists(ROOT_DIR / '.env'):
    load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ.get('MONGO_URL')
if not mongo_url:
    raise ValueError("MONGO_URL environment variable is required")

client = AsyncIOMotorClient(mongo_url)
db_name = os.environ.get('DB_NAME', 'pdf_rag_db')
db = client[db_name]

# Initialize ChromaDB
environment = os.environ.get('ENVIRONMENT', 'local')
if environment == 'docker':
    # Use ChromaDB HTTP client to connect to ChromaDB service
    chroma_host = os.environ.get('CHROMA_HOST', 'chromadb')
    chroma_port = int(os.environ.get('CHROMA_PORT', 8000))
    chroma_client = chromadb.HttpClient(
        host=chroma_host,
        port=chroma_port
    )
else:
    # For local development with containerized ChromaDB
    chroma_host = os.environ.get('CHROMA_HOST', 'localhost')
    chroma_port = int(os.environ.get('CHROMA_PORT', 8000))
    try:
        # Try HTTP client first (for containerized ChromaDB)
        chroma_client = chromadb.HttpClient(
            host=chroma_host,
            port=chroma_port
        )
        # Test the connection
        chroma_client.heartbeat()
    except Exception as e:
        logging.warning(f"Could not connect to ChromaDB HTTP service: {e}")
        # Fallback to local persistent client
        chroma_db_path = os.environ.get('CHROMA_DB_PATH', './chroma_db')
        chroma_client = chromadb.PersistentClient(path=chroma_db_path)

# Initialize sentence transformer for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# OpenAI/OpenRouter LLM setup
openai_api_key = os.environ.get('OPENAI_API_KEY')
if openai_api_key:
    # Check if it's an OpenRouter key (starts with sk-or-v1)
    if openai_api_key.startswith('sk-or-v1'):
        openai_client = AsyncOpenAI(
            api_key=openai_api_key,
            base_url="https://openrouter.ai/api/v1"
        )
    else:
        openai_client = AsyncOpenAI(api_key=openai_api_key)
else:
    openai_client = None

# Document Models
class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    content: str
    upload_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    chunk_ids: List[str] = Field(default=[])

class DocumentCreate(BaseModel):
    filename: str
    content: str

class DocumentChunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    content: str
    chunk_index: int
    embedding: List[float] = Field(default=[])

# Chat Models
class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    message: str
    response: str
    source_chunks: List[Dict[str, Any]] = Field(default=[])
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    source_documents: List[Dict[str, str]] = Field(default=[])

# Status Check Models
class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StatusCheckCreate(BaseModel):
    client_name: str

# Helper Functions
def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into chunks with overlap."""
    chunks = []
    words = text.split()
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words)
        if chunk.strip():
            chunks.append(chunk)
        
        if i + chunk_size >= len(words):
            break
    
    return chunks

async def store_document_chunks(document_id: str, chunks: List[str]):
    """Store document chunks in ChromaDB and MongoDB."""
    try:
        # Get or create collection in ChromaDB
        collection_name = "document_chunks"
        try:
            collection = chroma_client.get_collection(collection_name)
        except:
            collection = chroma_client.create_collection(collection_name)
        
        chunk_ids = []
        for i, chunk_content in enumerate(chunks):
            chunk_id = f"{document_id}_chunk_{i}"
            
            # Generate embedding
            embedding = embedding_model.encode(chunk_content).tolist()
            
            # Store in ChromaDB
            collection.add(
                documents=[chunk_content],
                embeddings=[embedding],
                ids=[chunk_id],
                metadatas=[{"document_id": document_id, "chunk_index": i}]
            )
            
            # Store in MongoDB
            chunk_doc = DocumentChunk(
                id=chunk_id,
                document_id=document_id,
                content=chunk_content,
                chunk_index=i,
                embedding=embedding
            )
            await db.document_chunks.insert_one(chunk_doc.dict())
            chunk_ids.append(chunk_id)
        
        return chunk_ids
    except Exception as e:
        logging.error(f"Error storing chunks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error storing document chunks: {str(e)}")

async def search_relevant_chunks(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search for relevant document chunks using semantic similarity."""
    try:
        # Get collection from ChromaDB
        collection = chroma_client.get_collection("document_chunks")
        
        # Generate query embedding
        query_embedding = embedding_model.encode(query).tolist()
        
        # Search for similar chunks
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        relevant_chunks = []
        if results['documents']:
            for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                relevant_chunks.append({
                    "content": doc,
                    "document_id": metadata['document_id'],
                    "chunk_index": metadata['chunk_index'],
                    "similarity_score": 1 - results['distances'][0][i] if results['distances'] else 0.5
                })
        
        return relevant_chunks
    except Exception as e:
        logging.error(f"Error searching chunks: {str(e)}")
        return []

async def generate_rag_response(query: str, context_chunks: List[Dict[str, Any]], chat_history: List[Dict[str, str]] = None) -> str:
    """Generate response using OpenAI LLM with retrieved context and conversation history."""
    try:
        if not openai_client:
            return "I apologize, but the AI service is currently not configured. Please contact your system administrator."

        # Prepare context from retrieved chunks
        context = "\n\n".join([f"Document excerpt {i+1}:\n{chunk['content']}"
                              for i, chunk in enumerate(context_chunks)])

        # Create messages for OpenAI Chat API
        messages = [
            {
                "role": "system",
                "content": """You are VTransact Corporate Assistant, a professional AI chatbot designed to help employees access and understand corporate documents.

Your role:
- Provide accurate, professional responses based on corporate documentation
- Use only information from the provided document context
- Maintain a helpful, corporate-appropriate tone
- Reference specific documents when citing information
- If information is not available in the provided context, politely inform the user
- Maintain conversation context to assist with follow-up questions

Be concise, accurate, and professional in all responses."""
            }
        ]

        # Add conversation history (last 5 exchanges to keep context manageable)
        if chat_history:
            for msg in chat_history[-5:]:
                messages.append({
                    "role": "user",
                    "content": msg["message"]
                })
                messages.append({
                    "role": "assistant",
                    "content": msg["response"]
                })

        # Add current query with context
        messages.append({
            "role": "user",
            "content": f"""Context from documents:
{context}

Question: {query}

Please answer the question based on the provided context."""
        })
        
        # Get response from OpenAI/OpenRouter
        # Use different model based on the API being used
        model = "openai/gpt-3.5-turbo" if openai_api_key.startswith('sk-or-v1') else "gpt-3.5-turbo"

        response = await openai_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1000,
            temperature=0.3,  # Lower temperature for more focused responses
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logging.error(f"Error generating OpenAI response: {str(e)}")
        return "I apologize, but I'm having trouble processing your request right now. Please try again later."

# Routes
@api_router.get("/")
async def root():
    return {"message": "RAG Chatbot API is running"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.dict())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]

@api_router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a PDF document."""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Extract text from PDF
        text_content = extract_text_from_pdf(file_content)
        
        if not text_content.strip():
            raise HTTPException(status_code=400, detail="No text content found in PDF")
        
        # Create document record
        document = Document(
            filename=file.filename,
            content=text_content
        )
        
        # Chunk the text
        chunks = chunk_text(text_content)
        
        # Store chunks in vector database
        chunk_ids = await store_document_chunks(document.id, chunks)
        document.chunk_ids = chunk_ids
        
        # Store document in MongoDB
        await db.documents.insert_one(document.dict())
        
        return {
            "document_id": document.id,
            "filename": file.filename,
            "chunks_created": len(chunks),
            "message": "Document uploaded and processed successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@api_router.get("/documents", response_model=List[Dict[str, Any]])
async def get_documents():
    """Get all uploaded documents."""
    try:
        documents = await db.documents.find().to_list(1000)
        return [
            {
                "id": doc["id"],
                "filename": doc["filename"],
                "upload_time": doc["upload_time"],
                "chunks_count": len(doc.get("chunk_ids", []))
            }
            for doc in documents
        ]
    except Exception as e:
        logging.error(f"Error fetching documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching documents")

@api_router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its chunks."""
    try:
        # Find document
        document = await db.documents.find_one({"id": document_id})
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete chunks from ChromaDB
        collection = chroma_client.get_collection("document_chunks")
        chunk_ids = document.get("chunk_ids", [])
        if chunk_ids:
            collection.delete(ids=chunk_ids)
        
        # Delete chunks from MongoDB
        await db.document_chunks.delete_many({"document_id": document_id})
        
        # Delete document from MongoDB
        await db.documents.delete_one({"id": document_id})
        
        return {"message": "Document deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail="Error deleting document")

@api_router.post("/chat", response_model=ChatResponse)
async def chat_with_documents(request: ChatRequest):
    """Chat with uploaded documents using RAG."""
    try:
        session_id = request.session_id or str(uuid.uuid4())

        # Get chat history for this session
        chat_history = await db.chat_messages.find(
            {"session_id": session_id}
        ).sort("timestamp", 1).to_list(100)

        # Search for relevant document chunks
        relevant_chunks = await search_relevant_chunks(request.message, top_k=5)

        if not relevant_chunks:
            return ChatResponse(
                response="I apologize, but I don't have access to any relevant corporate documents to answer your question. Please ensure the necessary documents have been uploaded to the system.",
                session_id=session_id,
                source_documents=[]
            )

        # Generate response using RAG with conversation history
        response_text = await generate_rag_response(request.message, relevant_chunks, chat_history)
        
        # Prepare source documents info
        source_documents = []
        unique_docs = {}
        for chunk in relevant_chunks:
            doc_id = chunk['document_id']
            if doc_id not in unique_docs:
                # Get document info
                doc = await db.documents.find_one({"id": doc_id})
                if doc:
                    unique_docs[doc_id] = {
                        "document_id": doc_id,
                        "filename": doc['filename'],
                        "relevance_score": f"{chunk['similarity_score']:.2f}"
                    }
        
        source_documents = list(unique_docs.values())
        
        # Store chat message
        chat_message = ChatMessage(
            session_id=session_id,
            message=request.message,
            response=response_text,
            source_chunks=[{
                "document_id": chunk['document_id'],
                "chunk_index": chunk['chunk_index'],
                "similarity_score": chunk['similarity_score']
            } for chunk in relevant_chunks]
        )
        
        await db.chat_messages.insert_one(chat_message.dict())
        
        return ChatResponse(
            response=response_text,
            session_id=session_id,
            source_documents=source_documents
        )
    
    except Exception as e:
        logging.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@api_router.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    """Get chat history for a session."""
    try:
        messages = await db.chat_messages.find(
            {"session_id": session_id}
        ).sort("timestamp", 1).to_list(1000)
        
        return [
            {
                "message": msg["message"],
                "response": msg["response"],
                "timestamp": msg["timestamp"],
                "source_documents": [
                    (await db.documents.find_one({"id": chunk["document_id"]}))["filename"]
                    for chunk in msg.get("source_chunks", [])
                    if await db.documents.find_one({"id": chunk["document_id"]})
                ][:3]  # Limit to 3 source documents for display
            }
            for msg in messages
        ]
    
    except Exception as e:
        logging.error(f"Error fetching chat history: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching chat history")

@api_router.get("/chat/sessions")
async def get_chat_sessions():
    """Get all chat sessions."""
    try:
        pipeline = [
            {"$group": {
                "_id": "$session_id",
                "last_message_time": {"$max": "$timestamp"},
                "message_count": {"$sum": 1}
            }},
            {"$sort": {"last_message_time": -1}},
            {"$limit": 50}
        ]
        
        sessions = await db.chat_messages.aggregate(pipeline).to_list(50)
        
        return [
            {
                "session_id": session["_id"],
                "last_message_time": session["last_message_time"],
                "message_count": session["message_count"]
            }
            for session in sessions
        ]
    
    except Exception as e:
        logging.error(f"Error fetching chat sessions: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching chat sessions")

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

# Test OpenAI connection on startup (commented out for development)
# @app.on_event("startup")
# async def test_openai_connection():
#     try:
#         if not openai_client:
#             logger.warning("OpenAI API key is not configured")
#             return
#         
#         # Simple test to verify OpenAI connection
#         response = await openai_client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[{"role": "user", "content": "Say 'OpenAI connection successful'"}],
#             max_tokens=10
#         )
#         logger.info("OpenAI connection test successful")
#     except Exception as e:
#         logger.error(f"OpenAI connection test failed: {str(e)}")