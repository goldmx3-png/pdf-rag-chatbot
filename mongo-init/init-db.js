// Initialize PDF RAG database
db = db.getSiblingDB('pdf_rag_db');

// Create collections with proper indexing
db.createCollection('documents');
db.createCollection('document_chunks');
db.createCollection('chat_messages');
db.createCollection('status_checks');

// Create indexes for better performance
db.documents.createIndex({ "id": 1 });
db.documents.createIndex({ "filename": 1 });
db.documents.createIndex({ "upload_time": 1 });

db.document_chunks.createIndex({ "id": 1 });
db.document_chunks.createIndex({ "document_id": 1 });
db.document_chunks.createIndex({ "chunk_index": 1 });

db.chat_messages.createIndex({ "id": 1 });
db.chat_messages.createIndex({ "session_id": 1 });
db.chat_messages.createIndex({ "timestamp": 1 });

db.status_checks.createIndex({ "id": 1 });
db.status_checks.createIndex({ "client_name": 1 });
db.status_checks.createIndex({ "timestamp": 1 });

print('PDF RAG database initialized successfully!');