#!/usr/bin/env python3
"""
Script to rebuild ChromaDB from existing MongoDB documents
"""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Load environment variables
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Initialize ChromaDB with persistence
chroma_client = chromadb.PersistentClient(path="/app/chroma_db")

# Initialize sentence transformer for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

async def rebuild_chromadb():
    """Rebuild ChromaDB from MongoDB documents"""
    try:
        print("🔄 Starting ChromaDB rebuild...")
        
        # Get or create collection
        collection_name = "document_chunks"
        try:
            collection = chroma_client.get_collection(collection_name)
            print(f"📁 Found existing collection: {collection_name}")
            # Clear existing data
            collection.delete()
            print("🗑️ Cleared existing collection data")
        except:
            print(f"📁 Collection {collection_name} does not exist, will create new one")
        
        # Create new collection
        collection = chroma_client.create_collection(collection_name)
        
        # Get all document chunks from MongoDB
        chunks = await db.document_chunks.find().to_list(10000)
        print(f"📄 Found {len(chunks)} document chunks in MongoDB")
        
        if not chunks:
            print("⚠️ No document chunks found in MongoDB")
            return
        
        # Batch process chunks
        batch_size = 100
        total_processed = 0
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Prepare batch data
            documents = []
            embeddings = []
            ids = []
            metadatas = []
            
            for chunk in batch:
                chunk_id = chunk['id']
                content = chunk['content']
                document_id = chunk['document_id']
                chunk_index = chunk['chunk_index']
                
                # Generate embedding if not available or empty
                if 'embedding' in chunk and chunk['embedding']:
                    embedding = chunk['embedding']
                else:
                    print(f"🔄 Generating embedding for chunk {chunk_id}")
                    embedding = embedding_model.encode(content).tolist()
                    
                    # Update MongoDB with the embedding
                    await db.document_chunks.update_one(
                        {"id": chunk_id},
                        {"$set": {"embedding": embedding}}
                    )
                
                documents.append(content)
                embeddings.append(embedding)
                ids.append(chunk_id)
                metadatas.append({
                    "document_id": document_id,
                    "chunk_index": chunk_index
                })
            
            # Add batch to ChromaDB
            collection.add(
                documents=documents,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas
            )
            
            total_processed += len(batch)
            print(f"✅ Processed {total_processed}/{len(chunks)} chunks")
        
        print(f"🎉 ChromaDB rebuild completed! Processed {total_processed} chunks")
        
        # Test the collection
        test_results = collection.query(
            query_texts=["test query"],
            n_results=1
        )
        
        if test_results['documents'] and len(test_results['documents'][0]) > 0:
            print(f"✅ ChromaDB test successful - found {len(test_results['documents'][0])} results")
        else:
            print("⚠️ ChromaDB test returned no results")
            
    except Exception as e:
        print(f"❌ Error rebuilding ChromaDB: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Close MongoDB connection
        client.close()

if __name__ == "__main__":
    asyncio.run(rebuild_chromadb())