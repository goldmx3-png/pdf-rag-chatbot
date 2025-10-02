# PDF RAG Chatbot

A full-stack intelligent chatbot application that uses Retrieval-Augmented Generation (RAG) to answer questions based on uploaded PDF documents. The system combines document retrieval with Large Language Models to provide accurate, context-aware responses.

## Features

- **📄 PDF Document Upload**: Upload and process PDF documents for analysis
- **🤖 Intelligent Chat**: Ask questions about your documents using natural language
- **🧠 Conversation Memory**: Maintains context across multiple conversation turns
- **🔍 Vector Search**: Uses ChromaDB for efficient semantic search
- **📊 Source Citations**: Shows which documents were used to generate responses
- **💬 Session Management**: Multiple chat sessions with history tracking
- **🎨 Modern UI**: Clean, responsive interface built with React and Tailwind CSS

## Tech Stack

### Backend
- **FastAPI**: High-performance Python web framework
- **MongoDB**: Document storage and chat history
- **ChromaDB**: Vector database for semantic search
- **Sentence Transformers**: Document embedding generation
- **OpenAI GPT**: Language model for response generation
- **PyPDF2**: PDF text extraction

### Frontend
- **React**: UI framework
- **Tailwind CSS**: Styling
- **Axios**: API communication
- **Lucide React**: Icons

### Infrastructure
- **Docker**: Containerization for MongoDB and ChromaDB
- **Docker Compose**: Multi-container orchestration

## Prerequisites

Before starting, ensure you have installed:

- **Docker & Docker Compose**: [Install Docker](https://docs.docker.com/get-docker/)
- **Python 3.8+**: [Install Python](https://www.python.org/downloads/)
- **Node.js 18+**: [Install Node.js](https://nodejs.org/)
- **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/account/api-keys) or [OpenRouter](https://openrouter.ai/)

## Quick Start Guide

### Step 1: Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd pdf-rag-chatbot

# Create environment file from template
cp .env.example .env
```

### Step 2: Configure Environment Variables

Edit `.env` and add your OpenAI API key:

```bash
# Open .env file and update:
OPENAI_API_KEY=your_actual_api_key_here
```

### Step 3: Start Docker Services

Start MongoDB and ChromaDB containers:

```bash
docker compose -f docker-compose.dev.yml up -d
```

Verify services are running:
```bash
docker ps
```

You should see:
- `pdf-rag-mongodb-dev` (port 27017)
- `pdf-rag-chromadb-dev` (port 8000)

### Step 4: Setup Backend

```bash
# Navigate to backend directory
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Copy environment variables to backend
cp ../.env .env

# Start the backend server
uvicorn server:app --reload --port 8080
```

Backend will be available at `http://localhost:8080`

### Step 5: Setup Frontend

Open a new terminal:

```bash
# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
npm install --legacy-peer-deps

# Create frontend environment file
echo "REACT_APP_BACKEND_URL=http://localhost:8080" > .env
echo "NODE_ENV=development" >> .env

# Start the development server
npm start
```

Frontend will automatically open at `http://localhost:3000`

## Using the Application

### 1. Upload Documents
- Click the **"Upload Document"** tab
- Select a PDF file (recommended max 10MB)
- Wait for processing to complete
- Document will appear in the documents list

### 2. Chat with Documents
- Switch to the **"Chat"** tab
- Type your question about the uploaded documents
- The chatbot will:
  - Search for relevant content
  - Generate a context-aware response
  - Show source documents used

### 3. Conversation Features
- **Follow-up questions**: The bot remembers previous messages in the session
- **New session**: Click to start a fresh conversation
- **Session history**: View past conversations from the dropdown

## Project Structure

```
pdf-rag-chatbot/
├── backend/                 # FastAPI backend
│   ├── server.py           # Main application file
│   ├── requirements.txt    # Python dependencies
│   ├── .env               # Backend environment (create from root .env)
│   └── chroma_db/         # Local ChromaDB storage (generated)
│
├── frontend/               # React frontend
│   ├── src/
│   │   ├── App.js         # Main React component
│   │   └── components/    # UI components
│   ├── package.json       # Node dependencies
│   └── .env              # Frontend environment
│
├── mongo-init/            # MongoDB initialization scripts
├── docker-compose.dev.yml # Development Docker services
├── .env                   # Root environment variables (create from .env.example)
├── .env.example          # Environment template
└── README.md             # This file
```

## Environment Variables Reference

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | OpenAI or OpenRouter API key | - | ✅ Yes |
| `MONGO_URL` | MongoDB connection string | `mongodb://admin:password123@localhost:27017/pdf_rag_db?authSource=admin` | ✅ Yes |
| `DB_NAME` | MongoDB database name | `pdf_rag_db` | ✅ Yes |
| `CHROMA_HOST` | ChromaDB host | `localhost` | ✅ Yes |
| `CHROMA_PORT` | ChromaDB port | `8000` | ✅ Yes |
| `ENVIRONMENT` | Runtime environment | `local` | No |
| `REACT_APP_BACKEND_URL` | Backend URL for frontend | `http://localhost:8080` | ✅ Yes |

## API Endpoints

### Documents
- `POST /api/documents/upload` - Upload a PDF document
- `GET /api/documents` - List all uploaded documents
- `DELETE /api/documents/{document_id}` - Delete a document

### Chat
- `POST /api/chat` - Send a message and get AI response
- `GET /api/chat/history/{session_id}` - Get chat history for a session
- `GET /api/chat/sessions` - List all chat sessions

### Health
- `GET /api/` - API health check

## Troubleshooting

### Port Already in Use

If you get port conflicts:

**Backend (Port 8080)**:
```bash
# Change backend port
uvicorn server:app --reload --port 8081

# Update frontend .env
REACT_APP_BACKEND_URL=http://localhost:8081
```

**Frontend (Port 3000)**:
```bash
# Frontend will prompt to use different port automatically
```

### Docker Services Not Starting

```bash
# Check Docker is running
docker ps

# Restart Docker services
docker compose -f docker-compose.dev.yml down
docker compose -f docker-compose.dev.yml up -d

# View logs
docker logs pdf-rag-mongodb-dev
docker logs pdf-rag-chromadb-dev
```

### OpenAI API Errors

**401 Unauthorized**:
- Verify your API key is correct in `.env` and `backend/.env`
- Check API key has sufficient credits
- Ensure key format matches provider (OpenAI vs OpenRouter)

**The bot automatically detects**:
- OpenAI keys: Start with `sk-proj-` or `sk-`
- OpenRouter keys: Start with `sk-or-v1-`

### Frontend Can't Connect to Backend

```bash
# Verify backend is running
curl http://localhost:8080/api/

# Check frontend environment
cat frontend/.env

# Restart frontend after env changes
cd frontend
npm start
```

### Python Dependencies Issues

```bash
# Downgrade numpy if needed
pip install "numpy<2"

# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall
```

## Stopping the Application

### Stop Backend and Frontend
Press `Ctrl+C` in their respective terminals

### Stop Docker Services
```bash
docker compose -f docker-compose.dev.yml down
```

### Complete Cleanup (removes data)
```bash
# Stop and remove all containers, networks, volumes
docker compose -f docker-compose.dev.yml down -v
```

## Production Deployment

For production deployment:

1. Use the main docker-compose.yml:
```bash
docker compose up -d
```

2. Set secure environment variables
3. Use HTTPS with proper SSL certificates
4. Configure authentication and authorization
5. Set up monitoring and logging

## Performance & Scalability

- **Document Chunking**: 500 chars with 100 char overlap
- **Vector Search**: Top-5 most relevant chunks per query
- **Conversation Memory**: Last 5 message exchanges
- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)

## Security Best Practices

- ✅ Never commit `.env` files with actual API keys
- ✅ Use environment variables for all sensitive data
- ✅ The `.gitignore` excludes all sensitive files
- ✅ In production, implement proper authentication
- ✅ Use HTTPS in production environments

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| "Connection refused" to MongoDB | Ensure Docker containers are running: `docker ps` |
| Frontend shows "Network Error" | Check REACT_APP_BACKEND_URL in frontend/.env |
| "Module not found" in Python | Install dependencies: `pip install -r requirements.txt` |
| ChromaDB not connecting | Verify ChromaDB container: `docker logs pdf-rag-chromadb-dev` |
| Slow embeddings | First run downloads ML model, subsequent runs are faster |

## Support & Resources

- **Issues**: Report bugs on GitHub Issues
- **Documentation**: Check `/docs` folder for additional guides
- **API Docs**: Visit `http://localhost:8080/docs` when backend is running

## License

[Add your license here]

---

**Built with ❤️ using FastAPI, React, and AI**
