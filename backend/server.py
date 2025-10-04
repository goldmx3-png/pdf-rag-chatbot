from fastapi import FastAPI, APIRouter, HTTPException, File, UploadFile, Form
from fastapi.responses import StreamingResponse
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
import httpx

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

# LLM setup - Create clients for both Ollama and OpenAI
llm_provider = os.environ.get('LLM_PROVIDER', 'ollama').lower()
ollama_base_url = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
ollama_model = os.environ.get('OLLAMA_MODEL', 'llama3.2:3b')

# Initialize Ollama client
ollama_client = AsyncOpenAI(
    api_key="ollama",  # Ollama doesn't require an API key
    base_url=f"{ollama_base_url}/v1"
)

# Initialize OpenAI client
openai_api_key = os.environ.get('OPENAI_API_KEY')
if openai_api_key:
    # Check if it's an OpenRouter key (starts with sk-or-v1)
    if openai_api_key.startswith('sk-or-v1'):
        openai_client = AsyncOpenAI(
            api_key=openai_api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        default_openai_model = "openai/gpt-3.5-turbo"
    else:
        openai_client = AsyncOpenAI(api_key=openai_api_key)
        default_openai_model = "gpt-3.5-turbo"
else:
    openai_client = None
    default_openai_model = None

# Set default model based on provider
if llm_provider == 'ollama':
    llm_model = ollama_model
else:
    llm_model = default_openai_model

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
    model: Optional[str] = None

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

async def generate_rag_response(query: str, context_chunks: List[Dict[str, Any]], chat_history: List[Dict[str, str]] = None, stream: bool = False, selected_model: str = None):
    """Generate response using LLM (OpenAI or Ollama) with retrieved context and conversation history."""
    try:
        # Determine the model to use
        model_to_use = selected_model or llm_model

        # Determine which client to use based on model
        is_ollama = model_to_use in ['llama3.2:3b', 'qwen2.5:3b'] or ':' in model_to_use

        if is_ollama:
            llm_client = ollama_client
        else:
            llm_client = openai_client

        if not llm_client:
            if stream:
                async def error_stream():
                    yield "I apologize, but the AI service is currently not configured. Please contact your system administrator."
                return error_stream()
            else:
                return "I apologize, but the AI service is currently not configured. Please contact your system administrator."

        # Prepare context from retrieved chunks
        context = "\n\n".join([f"Document excerpt {i+1}:\n{chunk['content'][:500]}"  # Limit chunk size
                              for i, chunk in enumerate(context_chunks)])

        # Create messages for Chat API
        # Use shorter system prompt for Ollama to reduce token count
        system_prompt = """You are a helpful AI assistant. Answer questions based on the provided document context.

Instructions:
- Answer directly using all relevant information from the document excerpts
- If excerpts contain complementary information on the same topic, synthesize them into a complete answer
- Only ask for clarification if excerpts discuss completely different topics that cannot be reasonably combined
- Be concise and professional""" if llm_provider == 'ollama' else """You are VTransact Corporate Assistant, a professional AI chatbot designed to help employees access and understand corporate documents.

Your role:
- Provide accurate, professional responses based on corporate documentation
- Use information from all provided document excerpts to give complete answers
- Maintain a helpful, corporate-appropriate tone
- Reference specific documents when citing information
- If information is not available in the provided context, politely inform the user
- Maintain conversation context to assist with follow-up questions

IMPORTANT - Handling Multiple Document Excerpts:
- When multiple excerpts are provided, analyze if they discuss the same topic or different topics
- If they discuss the SAME topic or complementary aspects, combine the information into one coherent answer
- ONLY ask the user to choose if the excerpts discuss completely DIFFERENT, unrelated topics
- Example of when to combine: Excerpts about different features of the same product → combine into complete answer
- Example of when to ask: One excerpt about Product A, another about Product B → ask which product they want to know about
- Default to providing a complete answer rather than asking for clarification

Be concise, accurate, and professional in all responses."""

        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]

        # Add conversation history - limit to 2 exchanges for Ollama, 5 for others
        history_limit = 2 if llm_provider == 'ollama' else 5
        if chat_history:
            for msg in chat_history[-history_limit:]:
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

Please answer based on the provided context."""
        })

        # Get response from LLM (Ollama or OpenAI)
        # Use fewer tokens for Ollama to speed up response
        if is_ollama:
            max_tokens = 300
        else:
            max_tokens = 1000

        # If streaming is requested, return an async generator
        if stream:
            async def stream_response():
                try:
                    response_stream = await llm_client.chat.completions.create(
                        model=model_to_use,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=0.3,
                        timeout=120.0 if is_ollama else 30.0,
                        stream=True
                    )

                    async for chunk in response_stream:
                        if chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content

                except Exception as e:
                    logging.error(f"Error in streaming response: {str(e)}")
                    yield f"\n\nError: {str(e)}"

            return stream_response()
        else:
            # Non-streaming response (original behavior)
            response = await llm_client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.3,
                timeout=120.0 if is_ollama else 30.0,
            )

            return response.choices[0].message.content

    except Exception as e:
        logging.error(f"Error generating OpenAI response: {str(e)}")
        if stream:
            async def error_stream():
                yield "I apologize, but I'm having trouble processing your request right now. Please try again later."
            return error_stream()
        else:
            return "I apologize, but I'm having trouble processing your request right now. Please try again later."

# Routes
@api_router.get("/")
async def root():
    return {"message": "RAG Chatbot API is running"}

@api_router.get("/models")
async def get_available_models():
    """Get list of available LLM models."""
    try:
        available_models = []

        # Always add OpenAI models if API key is configured
        if os.environ.get('OPENAI_API_KEY'):
            api_key = os.environ.get('OPENAI_API_KEY')
            if api_key.startswith('sk-or-v1'):
                # OpenRouter models
                available_models.extend([
                    {"id": "openai/gpt-3.5-turbo", "name": "GPT-3.5 Turbo (OpenRouter)", "provider": "openrouter"},
                    {"id": "openai/gpt-4", "name": "GPT-4 (OpenRouter)", "provider": "openrouter"},
                ])
            else:
                # OpenAI models
                available_models.extend([
                    {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "provider": "openai"},
                    {"id": "gpt-4", "name": "GPT-4", "provider": "openai"},
                ])

        # Check for Ollama models
        try:
            ollama_url = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{ollama_url}/api/tags", timeout=5.0)
                if response.status_code == 200:
                    data = response.json()
                    for model in data.get('models', []):
                        model_name = model.get('name', '')
                        available_models.append({
                            "id": model_name,
                            "name": f"{model_name.split(':')[0].upper()} ({model_name.split(':')[1] if ':' in model_name else 'latest'})",
                            "provider": "ollama"
                        })
        except Exception as e:
            logging.warning(f"Could not fetch Ollama models: {str(e)}")

        return {"models": available_models, "default": llm_model}

    except Exception as e:
        logging.error(f"Error fetching models: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching available models")

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

@api_router.post("/chat")
async def chat_with_documents(request: ChatRequest):
    """Chat with uploaded documents using RAG with streaming support."""
    try:
        session_id = request.session_id or str(uuid.uuid4())

        # Get chat history for this session
        chat_history = await db.chat_messages.find(
            {"session_id": session_id}
        ).sort("timestamp", 1).to_list(100)

        # Search for relevant document chunks - use fewer for Ollama to avoid context overflow
        top_k = 2 if llm_provider == 'ollama' else 5
        relevant_chunks = await search_relevant_chunks(request.message, top_k=top_k)

        if not relevant_chunks:
            # For non-streaming, return error as JSON
            return ChatResponse(
                response="I apologize, but I don't have access to any relevant corporate documents to answer your question. Please ensure the necessary documents have been uploaded to the system.",
                session_id=session_id,
                source_documents=[]
            )

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

        # Generate streaming response
        async def event_generator():
            full_response = ""

            # Send session_id and source_documents first
            yield f"data: {json.dumps({'type': 'metadata', 'session_id': session_id, 'source_documents': source_documents})}\n\n"

            # Stream the response
            response_stream = await generate_rag_response(request.message, relevant_chunks, chat_history, stream=True, selected_model=request.model)

            async for chunk in response_stream:
                full_response += chunk
                yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"

            # Send done signal
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

            # Store chat message after streaming is complete
            chat_message = ChatMessage(
                session_id=session_id,
                message=request.message,
                response=full_response,
                source_chunks=[{
                    "document_id": chunk['document_id'],
                    "chunk_index": chunk['chunk_index'],
                    "similarity_score": chunk['similarity_score']
                } for chunk in relevant_chunks]
            )

            await db.chat_messages.insert_one(chat_message.dict())

        return StreamingResponse(event_generator(), media_type="text/event-stream")

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