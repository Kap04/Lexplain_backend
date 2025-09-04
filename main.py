import os
import json
from fastapi import FastAPI, Depends, HTTPException, Request, Body, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any, Tuple
import firebase_admin
from firebase_admin import auth, credentials
from google.cloud import firestore
from google.oauth2 import service_account
from dotenv import load_dotenv
# Add these imports at the top of main.py
import threading
import traceback
from datetime import datetime
import time
import numpy as np
import fitz  # PyMuPDF
import io
import re

# Import caching and token counting systems
from token_counter import get_token_counter
from caching_system import get_cache_system

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.document_subscribers: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, document_id: str = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        if document_id:
            if document_id not in self.document_subscribers:
                self.document_subscribers[document_id] = []
            self.document_subscribers[document_id].append(websocket)

    def disconnect(self, websocket: WebSocket, document_id: str = None):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if document_id and document_id in self.document_subscribers:
            if websocket in self.document_subscribers[document_id]:
                self.document_subscribers[document_id].remove(websocket)
            if not self.document_subscribers[document_id]:
                del self.document_subscribers[document_id]

    async def send_document_update(self, document_id: str, message: dict):
        if document_id in self.document_subscribers:
            disconnected = []
            for connection in self.document_subscribers[document_id]:
                try:
                    await connection.send_json(message)
                except:
                    disconnected.append(connection)
            
            # Clean up disconnected clients
            for conn in disconnected:
                self.disconnect(conn, document_id)

manager = ConnectionManager()

load_dotenv()

# FastAPI app instance
app = FastAPI()

# Global variable to hold the Firestore client
db = None

from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:3000",               # for local dev
    "https://lexplain-ebon.vercel.app",    # your Vercel frontend
    "https://lexplain-ebon.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize Firebase Admin SDK and Firestore
if not firebase_admin._apps:
    try:
        # Try to get credentials from environment variable (Railway/Render approach)
        # Use a different env var name to avoid conflicts with Google's default credential discovery
        google_creds_json = os.getenv("GOOGLE_CREDENTIALS_JSON")
        
        if google_creds_json:
            # Parse the JSON string from environment variable
            creds_dict = json.loads(google_creds_json)
            
            # Initialize Firebase Admin
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred)
            
            # Initialize Firestore with explicit credentials
            firestore_creds = service_account.Credentials.from_service_account_info(creds_dict)
            db = firestore.Client(credentials=firestore_creds, project=creds_dict['project_id'])
            
            print("✅ Firebase and Firestore initialized successfully with env credentials")
            
        else:
            # Fallback to default credentials (local development)
            firebase_admin.initialize_app()
            db = firestore.Client()
            print("✅ Firebase and Firestore initialized with default credentials")
            
    except Exception as e:
        print(f"❌ Failed to initialize Firebase/Firestore: {e}")
        raise RuntimeError(f"Firebase initialization failed: {e}")
else:
    # If Firebase is already initialized, just get Firestore client
    try:
        google_creds_json = os.getenv("GOOGLE_CREDENTIALS_JSON")
        if google_creds_json:
            creds_dict = json.loads(google_creds_json)
            firestore_creds = service_account.Credentials.from_service_account_info(creds_dict)
            db = firestore.Client(credentials=firestore_creds, project=creds_dict['project_id'])
        else:
            db = firestore.Client()
        print("✅ Firestore client created (Firebase already initialized)")
    except Exception as e:
        print(f"❌ Failed to create Firestore client: {e}")
        raise RuntimeError(f"Firestore client creation failed: {e}")

# Firebase token verification dependency
def verify_firebase_token(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        print(f"Auth failed: Missing or invalid Authorization header. Got: {auth_header}")
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    id_token = auth_header.split("Bearer ")[1]
    try:
        decoded_token = auth.verify_id_token(id_token)
        print(f"Decoded token: {decoded_token}")
        return decoded_token
    except Exception as e:
        print(f"Auth failed: Invalid Firebase ID token. Error: {e}")
        raise HTTPException(status_code=401, detail="Invalid Firebase ID token")

# Phase 3.3: WebSocket endpoint for real-time document processing status
@app.websocket("/ws/{document_id}")
async def websocket_endpoint(websocket: WebSocket, document_id: str):
    await manager.connect(websocket, document_id)
    try:
        # Send initial status
        doc_ref = db.collection(os.getenv("FIRESTORE_DOCUMENTS_COLLECTION", "documents")).document(document_id)
        doc_snapshot = doc_ref.get()
        
        if doc_snapshot.exists:
            doc_data = doc_snapshot.to_dict()
            status = doc_data.get('status', 'unknown')
            
            await websocket.send_json({
                "type": "status_update",
                "document_id": document_id,
                "status": status,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Keep connection alive and wait for updates
        while True:
            await websocket.receive_text()  # Keep connection alive
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, document_id)
        print(f"WebSocket disconnected for document {document_id}")
    except Exception as e:
        print(f"WebSocket error for document {document_id}: {e}")
        manager.disconnect(websocket, document_id)

# Import firestore functions AFTER db is initialized
from firestore_adapter import (
    add_document_metadata, update_document_status, add_chunks, add_summary, 
    get_summary_by_doc_id, get_chunks_by_doc_id, add_qa_session, 
    get_qa_sessions_by_user, get_qa_session_by_id, update_qa_session_messages, 
    update_qa_session_field, delete_qa_session
)
from pipeline import chunk_text, debug_simple_embedding_test, embed_text, embed_texts, generate_summary

@app.delete("/api/chat/session/{session_id}")
def delete_chat_session(session_id: str, user=Depends(verify_firebase_token)):
    session = get_qa_session_by_id(db, session_id)
    if not session or session.get("userId") != user["uid"]:
        raise HTTPException(status_code=404, detail="Session not found")
    delete_qa_session(db, session_id)
    return {"success": True}

# --- PDF Content Extraction Helpers (PyMuPDF + OCR) ---
import fitz  # PyMuPDF
import io
import tempfile
import os

def extract_pages_from_pdf_content(content) -> List[Dict[str, Any]]:
    """Extract text from PDF file bytes using PyMuPDF."""
    if not content:
        raise ValueError("Empty PDF content")
    if not isinstance(content, (bytes, bytearray)):
        raise ValueError("PDF extraction expects bytes")
    doc = fitz.open(stream=content, filetype="pdf")
    if doc.is_encrypted:
        try:
            doc.authenticate("")
        except Exception:
            doc.close()
            raise ValueError("PDF is encrypted")
    pages = []
    for i in range(len(doc)):
        p = doc[i]
        # Try different text extraction methods, ensuring we get strings
        text = ""
        try:
            text = p.get_text("text")
        except:
            try:
                # Fallback to blocks method, but extract text properly
                blocks = p.get_text("blocks")
                if isinstance(blocks, list):
                    text = "\n".join(block[4] if isinstance(block, (list, tuple)) and len(block) > 4 else str(block) for block in blocks)
                else:
                    text = str(blocks)
            except:
                try:
                    # Last resort: get_text without parameters
                    text = p.get_text() or ""
                except:
                    text = ""
        
        pages.append({"page": i+1, "text": text})
    doc.close()
    return pages

def truncate_content_for_firestore(content: str, max_bytes: int = 900000) -> str:
    """
    Truncate content to fit within Firestore document size limits
    
    Args:
        content: Text content to truncate
        max_bytes: Maximum size in bytes (default 900KB, leaving room for other fields)
    
    Returns:
        Truncated content
    """
    if not content:
        return content
    
    # Convert to bytes to check size
    content_bytes = content.encode('utf-8')
    
    if len(content_bytes) <= max_bytes:
        return content
    
    # Truncate to fit within limit
    # We'll cut at character boundaries to avoid encoding issues
    truncated_content = content[:max_bytes // 2]  # Conservative estimate
    
    # Try to find a good break point (end of sentence or paragraph)
    for break_char in ['\n\n', '\n', '. ', '? ', '! ']:
        last_break = truncated_content.rfind(break_char)
        if last_break > max_bytes // 4:  # Don't truncate too aggressively
            truncated_content = truncated_content[:last_break + len(break_char)]
            break
    
    # Add truncation notice
    truncated_content += "\n\n[Content truncated due to size limits. Full content processed for analysis.]"
    
    return truncated_content

# def extract_text_with_ocr_support(content, filename: str = "document.pdf") -> tuple[str, str]:
#     """
#     Extract text from PDF with OCR fallback support
    
#     Returns:
#         tuple: (extracted_text, extraction_method)
#     """
#     try:
#         # First, try the existing method
#         pages = extract_pages_from_pdf_content(content)
#         extracted_text = "\n\n".join(page["text"] for page in pages)
        
#         # Check if we got meaningful text
#         text_quality = len(extracted_text.strip().replace('\n', '').replace(' ', ''))
        
#         if text_quality > 100:  # Good text extraction
#             print(f"✅ Text-based extraction successful: {text_quality} characters")
#             return extracted_text, "text_based"
#         else:
#             print(f"⚠️ Poor text extraction ({text_quality} chars), trying OCR...")
#             # Fallback to OCR
#             return _extract_with_ocr_fallback(content, filename)
            
    # except Exception as e:
    #     print(f"❌ Text-based extraction failed: {e}, trying OCR...")
    #     # Fallback to OCR
    #     return _extract_with_ocr_fallback(content, filename)

# def _extract_with_ocr_fallback(content, filename: str) -> tuple[str, str]:
#     """
#     Extract text using OCR as fallback
#     """
#     try:
#         # Import OCR processor
#         #from ocr_processor import extract_text_with_ocr
        
#         # Save content to temporary file for OCR processing
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
#             temp_file.write(content)
#             temp_path = temp_file.name
        
#         try:
#             # Use OCR processor
#             extracted_text, method = extract_text_with_ocr(temp_path)
#             print(f"✅ OCR extraction successful: {len(extracted_text)} characters using {method}")
#             return extracted_text, f"ocr_{method}"
#         finally:
#             # Clean up temp file
#             try:
#                 os.unlink(temp_path)
#             except:
#                 pass
                
#     except Exception as e:
#         print(f"❌ OCR extraction also failed: {e}")
#         # Last resort: return as plain text
#         try:
#             text = content.decode('utf-8', errors='ignore')
#             return text, "plain_text_fallback"
#         except:
#             return "Could not extract text from document", "extraction_failed"


def _create_combined_summary(summary_data: Dict[str, Any]) -> str:
    """Create a combined summary text from bullets and risks."""
    combined_parts = []
    
    # Add document summary bullets
    bullets = summary_data.get("bullets", [])
    if bullets:
        combined_parts.append("📋 **Document Summary:**")
        for bullet in bullets:
            if bullet.strip():
                combined_parts.append(f"• {bullet}")
    
    # Add potential risks
    risks = summary_data.get("risks", [])
    if risks:
        combined_parts.append("\n⚠️ **Potential Risks & Considerations:**")
        for risk in risks:
            if isinstance(risk, dict) and risk.get("label") and risk.get("explanation"):
                combined_parts.append(f"• **{risk['label']}:** {risk['explanation']}")
            elif isinstance(risk, str):
                combined_parts.append(f"• {risk}")
    
    # If no bullets or risks, provide a fallback
    if not combined_parts:
        combined_parts = ["Document processed successfully, but no summary content was generated."]
    
    return "\n".join(combined_parts)


def _clean_message_for_response(message: Dict[str, Any]) -> Dict[str, Any]:
    """Clean a message object to remove Firestore sentinels and make it JSON-serializable."""
    cleaned = dict(message)
    
    # Replace SERVER_TIMESTAMP with current ISO timestamp
    if "timestamp" in cleaned:
        timestamp = cleaned["timestamp"]
        if str(type(timestamp)) == "<class 'google.cloud.firestore_v1.transforms.Sentinel'>":
            cleaned["timestamp"] = datetime.utcnow().isoformat()
        elif hasattr(timestamp, "isoformat"):
            cleaned["timestamp"] = timestamp.isoformat()
    
    return cleaned


# @app.post("/api/upload/content")
# async def upload_document_content(
#     file: UploadFile = File(...),
#     user=Depends(verify_firebase_token)
# ):
#     print(f"Uploading document: {file.filename} for user: {user['uid']}")
#     content = await file.read()
    
#     try:
#         # Use OCR-aware text extraction
#         extracted_text, extraction_method = extract_text_with_ocr_support(content, file.filename)
        
#         # Truncate content if too large for Firestore
#         original_length = len(extracted_text)
#         extracted_text = truncate_content_for_firestore(extracted_text)
        
#         if len(extracted_text) < original_length:
#             print(f"⚠️ Content truncated from {original_length} to {len(extracted_text)} characters for Firestore storage")
        
#         print(f"✅ Text extraction successful using {extraction_method}. Preview: {extracted_text[:200]}")
#     except Exception as e:
#         print(f"❌ All text extraction methods failed: {e}. Using fallback.")
#         extracted_text = "Error: Could not extract text from document"
#         extraction_method = "extraction_failed"

#     doc = {
#         "ownerId": user["uid"],
#         "filename": file.filename,
#         "status": "uploaded",
#         "createdAt": firestore.SERVER_TIMESTAMP,
#         "documentContent": extracted_text,
#         "extractionMethod": extraction_method,  # Track how text was extracted
#     }
#     doc_id = add_document_metadata(db, doc)  # Pass db
#     print(f"Document stored with ID: {doc_id}")

#     # --- Start processing in background so embeddings are created immediately ---
#     try:
#         t = threading.Thread(target=_process_document_sync, args=(doc_id, user["uid"]), daemon=True)
#         t.start()
#         print(f"Background processing thread started for {doc_id}")
#     except Exception as e:
#         print(f"Failed to start background thread for processing: {e}")

#     return {"document_id": doc_id, "status": "uploaded", "extraction_method": extraction_method}




def _process_document_sync(document_id: str, owner_uid: str):
    """Synchronous processing function that chunks, embeds, stores chunks and summary.
       Can be called directly (synchronously) or from a background thread.
    """
    def send_status_update(status: str, message: str = ""):
        """Send status update via WebSocket (syncwrapper)"""
        try:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, create a task
                    asyncio.create_task(manager.send_document_update(document_id, {
                        "type": "status_update",
                        "document_id": document_id,
                        "status": status,
                        "message": message,
                        "timestamp": datetime.utcnow().isoformat()
                    }))
                else:
                    # If no loop is running, run it
                    loop.run_until_complete(manager.send_document_update(document_id, {
                        "type": "status_update",
                        "document_id": document_id,
                        "status": status,
                        "message": message,
                        "timestamp": datetime.utcnow().isoformat()
                    }))
            except RuntimeError:
                # No event loop in current thread, skip WebSocket update
                print(f"📡 Status update: {status} - {message}")
        except Exception as e:
            print(f"Failed to send WebSocket update: {e}")
    
    try:
        # Send initial processing status
        send_status_update("processing", "Starting document processing...")
        
        # Add this before the embedding code in _process_document_sync
        debug_simple_embedding_test()
        print(f"[processor] Starting processing for {document_id} (owner {owner_uid})")
        update_document_status(db, document_id, "processing")  # Pass db
        
        # Send status update
        send_status_update("processing", "Extracting document content...")
        
        doc_ref = db.collection(os.getenv("FIRESTORE_DOCUMENTS_COLLECTION", "documents")).document(document_id)
        snapshot = doc_ref.get()
        if not snapshot.exists:
            print(f"[processor] Document {document_id} not found")
            update_document_status(db, document_id, "failed")  # Pass db
            send_status_update("failed", "Document not found")
            return

        doc = snapshot.to_dict() or {}
        content = doc.get("documentContent", "")
        if not content:
            print(f"[processor] No content for {document_id}")
            update_document_status(db, document_id, "failed")  # Pass db
            send_status_update("failed", "No document content found")
            return

        # create pages (you are storing as single-page extracted content)
        pages = [{"page": 1, "text": content}]
        chunks = chunk_text(pages)
        if not chunks:
            print(f"[processor] No chunks generated for {document_id}")
            update_document_status(db, document_id, "failed")  # Pass db
            send_status_update("failed", "Failed to process document text")
            return

        print(f"[processor] {len(chunks)} chunks created for {document_id} (preview: {chunks[0]['text'][:120]})")
        send_status_update("processing", f"Created {len(chunks)} text chunks, generating embeddings...")

        # embed in batches with delays
        batch_size = int(os.getenv("EMBED_BATCH_SIZE", 2))  # Reduced default batch size
        try:
            texts = [c["text"] for c in chunks]
            total_batches = (len(texts) + batch_size - 1) // batch_size
            print(f"[processor] Processing {len(texts)} texts in {total_batches} batches")
            
            for i in range(0, len(texts), batch_size):
                batch_num = (i // batch_size) + 1
                batch_texts = texts[i:i+batch_size]
                
                print(f"[processor] Processing embedding batch {batch_num}/{total_batches} ({len(batch_texts)} items)")
                send_status_update("processing", f"Processing embeddings batch {batch_num}/{total_batches}...")
                
                try:
                    # Add a small delay before each batch
                    if i > 0:  # Don't delay before the first batch
                        sleep_time = 3.0  # 3 seconds between batches
                        print(f"[processor] Waiting {sleep_time}s before next embedding batch...")
                        time.sleep(sleep_time)
                    
                    embeddings = embed_texts(batch_texts)
                    
                    # Assign embeddings to chunks
                    for j, emb in enumerate(embeddings):
                        idx = i + j
                        if idx < len(chunks):  # Safety check
                            chunks[idx]["embedding"] = emb
                            chunks[idx]["documentId"] = document_id
                    
                    print(f"[processor] ✅ Batch {batch_num}/{total_batches} completed successfully")
                    
                except Exception as batch_e:
                    print(f"[processor] ❌ Embedding error for batch {batch_num}: {batch_e}")
                    # Add longer delay before retrying or continuing
                    time.sleep(5.0)
                    raise batch_e
            
            print(f"[processor] ✅ All embeddings generated for {len(chunks)} chunks.")
            send_status_update("processing", "Embeddings complete, storing data...")
            
        except Exception as e:
            print(f"[processor] Embedding error for {document_id}: {e}")
            traceback.print_exc()
            update_document_status(db, document_id, "failed")  # Pass db
            send_status_update("failed", "Failed to generate embeddings")
            return

        # Small delay before storing chunks
        print("[processor] Waiting 2s before storing chunks...")
        time.sleep(2.0)

        # persist chunks and summary
        try:
            add_chunks(db, chunks)  # Pass db
            print(f"[processor] ✅ Chunks stored successfully")
            send_status_update("processing", "Generating document summary...")
        except Exception as e:
            print(f"[processor] add_chunks failed for {document_id}: {e}")
            traceback.print_exc()
            update_document_status(db, document_id, "failed")  # Pass db
            send_status_update("failed", "Failed to store document chunks")
            return

        # Small delay before generating summary
        print("[processor] Waiting 2s before generating summary...")
        time.sleep(2.0)

        try:
            print(f"[processor] Generating summary for {len(chunks)} chunks...")
            summary_data = generate_summary(chunks)
            
            # Create combined summary text
            combined_summary = _create_combined_summary(summary_data)
            
            summary_doc = {
                "documentId": document_id, 
                "bullets": summary_data.get("bullets", []),
                "risks": summary_data.get("risks", []),
                "summary": combined_summary  # Add the combined summary
            }
            print("Generated summary after processing:", summary_doc)
            add_summary(db, summary_doc)  # Pass db
            print(f"[processor] ✅ Summary stored successfully")
            
        except Exception as e:
            print(f"[processor] add_summary failed for {document_id}: {e}")
            traceback.print_exc()
            # still mark processed if chunking/embeds worked; but mark partial
            update_document_status(db, document_id, "processed_with_summary_error")  # Pass db
            send_status_update("processed", "Document processed (summary generation had issues)")
            return

        update_document_status(db, document_id, "processed")  # Pass db
        print(f"[processor] ✅ Document {document_id} processed successfully.")
        send_status_update("processed", "Document processing complete!")
        
    except Exception as e:
        print(f"[processor] Unexpected error while processing {document_id}: {e}")
        traceback.print_exc()
        try:
            update_document_status(db, document_id, "failed")  # Pass db
            send_status_update("failed", f"Processing failed: {str(e)}")
        except Exception:
            pass

@app.post("/api/process/{document_id}")
def process_document(document_id: str, user=Depends(verify_firebase_token)):
    # Synchronous request to (re-)process the document using same helper:
    _process_document_sync(document_id, user["uid"])
    return {"status": "processing_started"}

@app.get("/api/documents/{document_id}/summary")
def get_summary(document_id: str, user=Depends(verify_firebase_token)):
    summary_data = get_summary_by_doc_id(db, document_id)  # Pass db
    if not summary_data:
        raise HTTPException(status_code=404, detail="Summary not found")
    
    # If we don't have a combined summary field, create it on the fly
    if "summary" not in summary_data:
        combined_summary = _create_combined_summary(summary_data)
        summary_data["summary"] = combined_summary
    
    print("Fetched summary of the document:", summary_data)
    return {**summary_data, "document_id": document_id}

# --- Similarity Search Helpers ---
import numpy as np

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def mmr(query_emb, chunk_embs, K=8, lambda_=0.7):
    selected = []
    candidate_idxs = list(range(len(chunk_embs)))
    sim_to_query = [cosine_similarity(query_emb, emb) for emb in chunk_embs]
    while len(selected) < K and candidate_idxs:
        if not selected:
            idx = max(candidate_idxs, key=lambda i: sim_to_query[i])
            selected.append(idx)
            candidate_idxs.remove(idx)
        else:
            scores = []
            for i in candidate_idxs:
                diversity = max([cosine_similarity(chunk_embs[i], chunk_embs[j]) for j in selected]) if selected else 0
                score = lambda_ * sim_to_query[i] - (1 - lambda_) * diversity
                scores.append((score, i))
            idx = max(scores)[1]
            selected.append(idx)
            candidate_idxs.remove(idx)
    return selected

@app.post("/api/documents/{document_id}/query")
def query_document(document_id: str, data: dict = Body(...), user=Depends(verify_firebase_token)):
    question = data.get("question")
    chunks = get_chunks_by_doc_id(db, document_id)  # Pass db
    chunk_texts = [c["text"] for c in chunks]
    chunk_embs = [c["embedding"] for c in chunks]
    # 1. Embed the query
    query_emb = embed_text(question)
    # 2. Top-K pool
    sim_scores = [cosine_similarity(query_emb, emb) for emb in chunk_embs]
    pool_size = min(50, len(chunks))
    top_pool_idxs = sorted(range(len(sim_scores)), key=lambda i: sim_scores[i], reverse=True)[:pool_size]
    pool_embs = [chunk_embs[i] for i in top_pool_idxs]
    pool_texts = [chunk_texts[i] for i in top_pool_idxs]
    # 3. MMR selection
    K = min(8, len(pool_embs))
    selected_idxs = mmr(query_emb, pool_embs, K=K, lambda_=0.7)
    selected_texts = [pool_texts[i] for i in selected_idxs]
    # 4. Gemini answer
    from google import genai
    from google.genai import types
    
    _API_KEY = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=_API_KEY)
    
    context = "\n".join(selected_texts)
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer in plain English in ≤ 120 words. If uncertain, respond 'I don't know — please consult a lawyer' and show the top 2 source snippets used."
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1
        )
    )
    answer = response.text if hasattr(response, 'text') else "No answer."
    sources = [
        {"document_id": document_id, "snippet": t[:60]} for t in selected_texts
    ]
    return {"answer": answer, "sources": sources}

@app.post("/api/documents/{document_id}/summarize")
def summarize_document(document_id: str, user=Depends(verify_firebase_token)):
    chunks = get_chunks_by_doc_id(db, document_id)  # Pass db
    if not chunks or len(chunks) == 0:
        print("No chunks found, processing document first...")
        process_document(document_id, user)
    chunks = get_chunks_by_doc_id(db, document_id)  # Pass db
    if not chunks:
        raise HTTPException(status_code=500, detail="Failed to generate chunks")

    chunk_texts = [c["text"] for c in chunks]
    chunk_embs = [c["embedding"] for c in chunks]
    # Use MMR to select diverse, representative chunks for summary
    summary_prompt = "Summarize the following document in plain English, focusing on key points and risks."
    summary_emb = embed_text(summary_prompt)
    pool_size = min(50, len(chunks))
    top_pool_idxs = sorted(range(len(chunk_embs)), key=lambda i: cosine_similarity(summary_emb, chunk_embs[i]), reverse=True)[:pool_size]
    pool_embs = [chunk_embs[i] for i in top_pool_idxs]
    pool_texts = [chunk_texts[i] for i in top_pool_idxs]
    K = min(10, len(pool_embs))
    selected_idxs = mmr(summary_emb, pool_embs, K=K, lambda_=0.5)
    selected_texts = [pool_texts[i] for i in selected_idxs]
    # Gemini summary
    from google import genai
    from google.genai import types
    
    _API_KEY = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=_API_KEY)
    
    context = "\n".join(selected_texts)
    prompt = f"Context: {context}\n{summary_prompt}"
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3
        )
    )
    summary = response.text if hasattr(response, 'text') else "No summary."
    print(f"Fetched {len(chunks)} chunks")
    print(f"Example chunk text: {chunks[0]['text'][:200] if chunks else 'None'}")

    return {"summary": summary}

@app.post("/api/documents/{document_id}/legal-analysis")
def generate_legal_analysis(document_id: str, user=Depends(verify_firebase_token)):
    """Generate comprehensive legal analysis for a document with Google Search integration."""
    chunks = get_chunks_by_doc_id(db, document_id)
    if not chunks or len(chunks) == 0:
        print("No chunks found, processing document first...")
        process_document(document_id, user)
    chunks = get_chunks_by_doc_id(db, document_id)
    if not chunks:
        raise HTTPException(status_code=500, detail="Failed to generate chunks")

    chunk_texts = [c["text"] for c in chunks]
    chunk_embs = [c["embedding"] for c in chunks]
    
    # Use MMR to select comprehensive content for analysis
    analysis_prompt = "Analyze this legal document for clauses, risks, and legal implications."
    analysis_emb = embed_text(analysis_prompt)
    pool_size = min(60, len(chunks))  # Larger pool for comprehensive analysis
    top_pool_idxs = sorted(range(len(chunk_embs)), key=lambda i: cosine_similarity(analysis_emb, chunk_embs[i]), reverse=True)[:pool_size]
    pool_embs = [chunk_embs[i] for i in top_pool_idxs]
    pool_texts = [chunk_texts[i] for i in top_pool_idxs]
    K = min(15, len(pool_embs))  # More chunks for detailed analysis
    selected_idxs = mmr(analysis_emb, pool_embs, K=K, lambda_=0.6)
    selected_texts = [pool_texts[i] for i in selected_idxs]
    
    context = "\n".join(selected_texts)
    
    # First, detect jurisdiction and document type
    jurisdiction_info = detect_jurisdiction_and_context(context)
    
    # Enhanced Gemini analysis with Google Search integration
    from google import genai
    from google.genai import types
    
    # Configure the client
    _API_KEY = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=_API_KEY)
    
    # # Define the grounding tool
    # grounding_tool = types.Tool(
    #     google_search=types.GoogleSearch()
    # )
    
    # # Configure generation settings
    # config = types.GenerateContentConfig(
    #     tools=[grounding_tool]
    # )
    
    legal_analysis_prompt = f"""
    You are a legal AI assistant with access to Google Search for current legal information.
    
    Document Context:
    {context}
    
    Detected Jurisdiction: {jurisdiction_info.get('jurisdiction', 'Unknown')}
    Document Type: {jurisdiction_info.get('document_type', 'Unknown')}
    
    Please analyze this legal document and provide comprehensive analysis. Use Google Search to:
    1. Find current legal requirements for this jurisdiction
    2. Check for recent legal precedents or changes
    3. Identify jurisdiction-specific compliance requirements
    4. Research best practices for this document type
    
    Provide your analysis in JSON format:
    {{
        "summary": "Brief summary including jurisdiction-specific context",
        "jurisdiction": "{{
            "detected": "{jurisdiction_info.get('jurisdiction', 'Unknown')}",
            "confidence": "high|medium|low",
            "applicable_laws": ["relevant laws found via search"],
            "recent_changes": ["recent legal changes affecting this document type"]
        }},
        "clauseCategories": [
            {{
                "category": "Category name",
                "clauses": ["List of specific clauses"],
                "riskLevel": "low|medium|high",
                "jurisdictionNotes": "Jurisdiction-specific considerations from search"
            }}
        ],
        "riskAnalysis": {{
            "overallRisk": "low|medium|high",
            "riskScore": 85,
            "highRiskClauses": [
                {{
                    "clause": "Specific high-risk clause text",
                    "risk": "Description of the risk",
                    "impact": "Potential impact",
                    "jurisdictionSpecific": "How this risk applies in detected jurisdiction"
                }}
            ],
            "complianceIssues": ["Potential compliance problems found via search"]
        }},
        "legalQuestions": [
            "Questions based on current legal requirements",
            "Jurisdiction-specific questions from search results"
        ],
        "searchInsights": [
            "Key insights from Google Search about this document type",
            "Recent legal developments affecting similar documents"
        ]
    }}

    Focus on:
    1. Current legal standards in the detected jurisdiction
    2. Recent case law or regulatory changes
    3. Industry-specific compliance requirements
    4. Best practices based on current legal guidance

    Respond ONLY with valid JSON.
    """
    
    try:
        # Make the request using the new Google Generative AI SDK with Google Search
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=legal_analysis_prompt,
            #config=config
        )
        
        analysis_text = response.text if hasattr(response, 'text') else "{}"
        
        # Check for grounding metadata (Google Search results)
        search_insights = []
        if hasattr(response, 'candidates') and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                grounding = candidate.grounding_metadata
                if hasattr(grounding, 'web_search_queries'):
                    search_insights.append(f"Searched for: {', '.join(grounding.web_search_queries)}")
                if hasattr(grounding, 'grounding_chunks'):
                    search_insights.append(f"Found {len(grounding.grounding_chunks)} relevant sources")
        
        # Clean the response to extract JSON
        analysis_text = analysis_text.strip()
        if analysis_text.startswith("```json"):
            analysis_text = analysis_text[7:]
        if analysis_text.endswith("```"):
            analysis_text = analysis_text[:-3]
        
        # Parse JSON response
        import json
        try:
            analysis_data = json.loads(analysis_text)
            
            # Add search insights if we have them
            if search_insights and 'searchInsights' in analysis_data:
                analysis_data['searchInsights'].extend(search_insights)
            elif search_insights:
                analysis_data['searchInsights'] = search_insights
                
        except json.JSONDecodeError:
            # Fallback structure if JSON parsing fails
            analysis_data = {
                "summary": "Analysis generated with search integration but format parsing failed.",
                "jurisdiction": {
                    "detected": jurisdiction_info.get('jurisdiction', 'Unknown'),
                    "confidence": "medium",
                    "applicable_laws": [],
                    "recent_changes": []
                },
                "clauseCategories": [
                    {
                        "category": "General Terms",
                        "clauses": ["Document contains various legal provisions"],
                        "riskLevel": "medium",
                        "jurisdictionNotes": "Manual review recommended"
                    }
                ],
                "riskAnalysis": {
                    "overallRisk": "medium",
                    "riskScore": 50,
                    "highRiskClauses": [],
                    "complianceIssues": []
                },
                "legalQuestions": [
                    "Please review this document with qualified legal counsel.",
                    "Are there jurisdiction-specific requirements to consider?",
                    "Should any terms be updated based on recent legal changes?"
                ],
                "searchInsights": ["Search integration available but parsing failed"]
            }
        
        print(f"Generated enhanced legal analysis with search for document {document_id}")
        return analysis_data
        
    except Exception as e:
        print(f"Error generating enhanced legal analysis: {e}")
        # Return a basic analysis structure
        return {
            "summary": "Unable to generate detailed analysis with search integration at this time.",
            "jurisdiction": {
                "detected": jurisdiction_info.get('jurisdiction', 'Unknown'),
                "confidence": "low",
                "applicable_laws": [],
                "recent_changes": []
            },
            "clauseCategories": [
                {
                    "category": "Document Review Required",
                    "clauses": ["Manual review recommended"],
                    "riskLevel": "medium",
                    "jurisdictionNotes": "Unable to determine jurisdiction-specific requirements"
                }
            ],
            "riskAnalysis": {
                "overallRisk": "medium",
                "riskScore": 50,
                "highRiskClauses": [],
                "complianceIssues": ["Unable to check compliance requirements"]
            },
            "legalQuestions": [
                "Please have this document reviewed by a qualified attorney.",
                "What jurisdiction applies to this document?",
                "Are there specific regulatory requirements to consider?"
            ],
            "searchInsights": ["Search integration temporarily unavailable"]
        }

def detect_jurisdiction_and_context(document_text):
    """Detect jurisdiction and document context from document text."""
    from google import genai
    from google.genai import types
    
    _API_KEY = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=_API_KEY)
    
    detection_prompt = f"""
    Analyze this legal document text and detect:
    1. Legal jurisdiction (country, state, province)
    2. Document type (contract, agreement, policy, etc.)
    3. Governing law clauses
    4. Company locations mentioned
    5. Currency or regional indicators
    
    Document text:
    {document_text[:3000]}  # Limit text for efficiency
    
    Respond in JSON format:
    {{
        "jurisdiction": "US-California" or "UK" or "EU-Germany" etc,
        "confidence": "high|medium|low",
        "document_type": "service_agreement" or "employment_contract" etc,
        "governing_law": "specific governing law mentioned",
        "indicators": ["specific phrases that indicate jurisdiction"]
    }}
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=detection_prompt
        )
        result_text = response.text if hasattr(response, 'text') else "{}"
        
        # Clean and parse
        result_text = result_text.strip()
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        
        import json
        return json.loads(result_text)
    except Exception as e:
        print(f"Error detecting jurisdiction: {e}")
        return {
            "jurisdiction": "Unknown",
            "confidence": "low",
            "document_type": "unknown",
            "governing_law": "not specified",
            "indicators": []
        }

@app.post("/api/documents/{document_id}/export-pdf")
def export_analysis_pdf(document_id: str, analysis_data: dict = Body(...), user=Depends(verify_firebase_token)):
    """Export legal analysis as PDF using fpdf."""
    try:
        from fpdf import FPDF
        import io
        from fastapi.responses import StreamingResponse
        import re
        
        # Helper function to clean text for PDF
        def clean_text_for_pdf(text):
            if not text:
                return ""
            # Replace Unicode characters with ASCII equivalents
            text = text.replace('\u2022', '-')  # bullet point
            text = text.replace('\u2013', '-')  # en dash
            text = text.replace('\u2014', '--') # em dash
            text = text.replace('\u2018', "'")  # left single quote
            text = text.replace('\u2019', "'")  # right single quote
            text = text.replace('\u201c', '"')  # left double quote
            text = text.replace('\u201d', '"')  # right double quote
            text = text.replace('\u2026', '...')  # ellipsis
            # Remove other non-ASCII characters
            text = re.sub(r'[^\x00-\x7F]+', '', text)
            return text
        
        # Helper function for text wrapping
        def add_wrapped_text(pdf, text, line_height=6, max_width=80):
            words = clean_text_for_pdf(text).split(' ')
            line = ""
            for word in words:
                if len(line + word) < max_width:
                    line += word + " "
                else:
                    if line.strip():
                        pdf.cell(0, line_height, line.strip(), ln=True)
                    line = word + " "
            if line.strip():
                pdf.cell(0, line_height, line.strip(), ln=True)
        
        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Title
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Legal Analysis Report", ln=True, align="C")
        pdf.ln(5)
        
        # Document info
        pdf.set_font("Arial", "B", 12)
        doc_name = clean_text_for_pdf(analysis_data.get('documentName', 'Legal Document'))
        pdf.cell(0, 10, f"Document: {doc_name}", ln=True)
        pdf.cell(0, 10, f"Document ID: {document_id}", ln=True)
        pdf.ln(5)
        
        # Summary
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Executive Summary", ln=True)
        pdf.set_font("Arial", size=10)
        summary_text = analysis_data.get('summary', 'No summary available')
        add_wrapped_text(pdf, summary_text)
        pdf.ln(5)
        
        # Risk Analysis
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Risk Assessment", ln=True)
        pdf.set_font("Arial", size=10)
        risk_analysis = analysis_data.get('riskAnalysis', {})
        pdf.cell(0, 6, f"Overall Risk Level: {risk_analysis.get('overallRisk', 'Unknown').upper()}", ln=True)
        pdf.cell(0, 6, f"Risk Score: {risk_analysis.get('riskScore', 0)}/100", ln=True)
        pdf.ln(5)
        
        # High Risk Clauses
        high_risk_clauses = risk_analysis.get('highRiskClauses', [])
        if high_risk_clauses:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "High Risk Clauses", ln=True)
            pdf.set_font("Arial", size=9)
            for i, risk_clause in enumerate(high_risk_clauses[:5], 1):  # Limit to 5 for space
                clause_text = clean_text_for_pdf(risk_clause.get('clause', ''))
                risk_text = clean_text_for_pdf(risk_clause.get('risk', ''))
                pdf.cell(0, 5, f"{i}. {clause_text[:60]}{'...' if len(clause_text) > 60 else ''}", ln=True)
                pdf.cell(0, 5, f"   Risk: {risk_text[:70]}{'...' if len(risk_text) > 70 else ''}", ln=True)
                pdf.ln(1)
            pdf.ln(3)
        
        # Clause Categories
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Clause Categories", ln=True)
        pdf.set_font("Arial", size=10)
        for category in analysis_data.get('clauseCategories', []):
            category_name = clean_text_for_pdf(category.get('category', 'Unknown'))
            risk_level = category.get('riskLevel', 'Unknown').upper()
            pdf.set_font("Arial", "B", 10)
            pdf.cell(0, 6, f"{category_name} (Risk: {risk_level})", ln=True)
            pdf.set_font("Arial", size=9)
            for clause in category.get('clauses', [])[:3]:  # Limit to 3 clauses per category
                clause_clean = clean_text_for_pdf(clause)
                if len(clause_clean) > 75:
                    clause_clean = clause_clean[:72] + "..."
                pdf.cell(0, 5, f"  - {clause_clean}", ln=True)
            pdf.ln(2)
        
        # Legal Questions
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Questions for Legal Counsel", ln=True)
        pdf.set_font("Arial", size=10)
        for i, question in enumerate(analysis_data.get('legalQuestions', [])[:5], 1):  # Limit to 5 questions
            question_clean = clean_text_for_pdf(question)
            if len(question_clean) > 75:
                question_clean = question_clean[:72] + "..."
            pdf.cell(0, 6, f"{i}. {question_clean}", ln=True)
        
        # Footer
        pdf.ln(10)
        pdf.set_font("Arial", "I", 8)
        pdf.cell(0, 6, "This analysis is generated by AI and should not replace professional legal advice.", ln=True)
        pdf.cell(0, 6, "Always consult with a qualified attorney for legal matters.", ln=True)
        
        # Return PDF as response
        pdf_output = pdf.output(dest='S')
        if isinstance(pdf_output, str):
            pdf_output = pdf_output.encode('latin1')
        
        return StreamingResponse(
            io.BytesIO(pdf_output),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={clean_text_for_pdf(analysis_data.get('documentName', 'legal_analysis'))}.pdf"}
        )
        
    except Exception as e:
        print(f"Error generating PDF: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")

@app.post("/api/documents/compare")
def compare_documents(data: dict = Body(...), user=Depends(verify_firebase_token)):
    """Compare multiple legal documents."""
    document_ids = data.get("document_ids", [])
    
    if len(document_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 documents required for comparison")
    
    try:
        # Check all documents are ready first
        document_statuses = []
        for doc_id in document_ids:
            try:
                doc_ref = db.collection(os.getenv("FIRESTORE_DOCUMENTS_COLLECTION", "documents")).document(doc_id)
                doc_snapshot = doc_ref.get()
                
                if not doc_snapshot.exists:
                    print(f"Document {doc_id} not found in Firestore")
                    continue
                    
                doc_data = doc_snapshot.to_dict()
                status = doc_data.get('status', 'unknown')
                document_statuses.append((doc_id, status, doc_data))
                print(f"Document {doc_id} status: {status}")
                
            except Exception as e:
                print(f"Error checking document {doc_id}: {e}")
                continue
        
        # If any documents are still processing, return a processing status
        processing_docs = [doc_id for doc_id, status, _ in document_statuses if status not in ['processed', 'processed_with_summary_error']]
        if processing_docs:
            return {
                "status": "processing",
                "message": f"Documents still processing: {', '.join(processing_docs)}",
                "ready_count": len(document_statuses) - len(processing_docs),
                "total_count": len(document_ids)
            }
        
        # Get analyses for all ready documents
        document_analyses = []
        for doc_id, status, doc_data in document_statuses:
            try:
                # Check if document is processed
                if doc_data.get('status') not in ['processed', 'processed_with_summary_error']:
                    print(f"Document {doc_id} not fully processed, skipping")
                    continue
                
                chunks = get_chunks_by_doc_id(db, doc_id)
                print(f"Found {len(chunks)} chunks for document {doc_id}")
                
                if not chunks:
                    print(f"No chunks found for document {doc_id}")
                    continue
                
                # Generate fresh analysis for comparison
                analysis = generate_document_analysis(doc_id, chunks)
                document_name = doc_data.get('filename', f'Document {len(document_analyses) + 1}')
                
                document_analyses.append({
                    "id": doc_id,
                    "name": document_name,
                    "analysis": analysis
                })
                print(f"Successfully analyzed document {doc_id}")
                
            except Exception as e:
                print(f"Error analyzing document {doc_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"Successfully analyzed {len(document_analyses)} out of {len(document_ids)} documents")
        
        if len(document_analyses) < 2:
            # Provide more helpful error message
            ready_docs = [doc_id for doc_id, status, _ in document_statuses if status in ['processed', 'processed_with_summary_error']]
            processing_docs = [doc_id for doc_id, status, _ in document_statuses if status not in ['processed', 'processed_with_summary_error']]
            
            error_msg = f"Need at least 2 processed documents for comparison. Ready: {len(ready_docs)}, Still processing: {len(processing_docs)}"
            if processing_docs:
                error_msg += f". Please wait for documents to finish processing: {', '.join(processing_docs)}"
            
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Generate comparison using Gemini
        comparison_result = generate_comparison_analysis(document_analyses)
        
        return {
            "id": f"comparison_{int(time.time())}",
            "documents": document_analyses,
            "comparison": comparison_result
        }
        
    except Exception as e:
        print(f"Error in document comparison: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to compare documents: {str(e)}")

def generate_document_analysis(doc_id, chunks):
    """Generate legal analysis for a single document with caching and token counting."""
    
    # Get instances
    cache_system = get_cache_system()
    token_counter = get_token_counter()
    
    # 1. Check if analysis is already cached
    cache_key = cache_system.generate_cache_key("analysis", doc_id, len(chunks))
    cached_result = cache_system.get_cached_result(cache_key)
    if cached_result:
        print(f"✅ Using cached analysis for document {doc_id}")
        return cached_result
    
    chunk_texts = [c["text"] for c in chunks]
    chunk_embs = [c["embedding"] for c in chunks]
    
    # 2. Create document cache for future use
    document_cache_name = cache_system.create_document_cache(doc_id, chunk_texts[:20], ttl_hours=2)
    
    # Use MMR to select comprehensive content for analysis
    analysis_prompt = "Analyze this legal document for clauses, risks, and legal implications."
    analysis_emb = embed_text(analysis_prompt)
    pool_size = min(40, len(chunks))
    top_pool_idxs = sorted(range(len(chunk_embs)), key=lambda i: cosine_similarity(analysis_emb, chunk_embs[i]), reverse=True)[:pool_size]
    pool_embs = [chunk_embs[i] for i in top_pool_idxs]
    pool_texts = [chunk_texts[i] for i in top_pool_idxs]
    K = min(10, len(pool_embs))
    selected_idxs = mmr(analysis_emb, pool_embs, K=K, lambda_=0.6)
    selected_texts = [pool_texts[i] for i in selected_idxs]
    
    context = "\n".join(selected_texts)
    
    analysis_prompt_full = f"""
    Analyze this legal document and extract structured information for comparison purposes.
    
    Document Content:
    {context}
    
    Provide analysis in JSON format:
    {{
        "summary": "Brief document summary",
        "clauseCategories": [
            {{
                "category": "Payment Terms",
                "clauses": ["specific clause texts"],
                "riskLevel": "low|medium|high"
            }}
        ],
        "riskScore": 75,
        "overallRisk": "medium",
        "keyTerms": ["important terms and conditions"],
        "jurisdiction": "detected legal jurisdiction if any",
        "documentType": "contract type or category"
    }}
    
    Focus on extracting comparable elements like payment terms, liability, termination, confidentiality, etc.
    """
    
    # 3. Count tokens before API call
    token_info = token_counter.count_tokens_before_request(analysis_prompt_full)
    print(f"📊 Analysis request - Estimated tokens: {token_info['input_tokens']}, Cost: ${token_info['estimated_cost_usd']:.4f}")
    
    # Generate structured analysis
    from google import genai
    from google.genai import types
    
    _API_KEY = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=_API_KEY)
    
    try:
        # 4. Use cached model if available
        generate_config = types.GenerateContentConfig(temperature=0.2)
        if document_cache_name:
            print(f"🗃️ Using document cache: {document_cache_name}")
            response = client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=[{"role": "user", "parts": [{"text": analysis_prompt_full}]}],
                config=generate_config,
                cached_content=document_cache_name
            )
        else:
            response = client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=analysis_prompt_full,
                config=generate_config
            )
        
        analysis_text = response.text if hasattr(response, 'text') else "{}"
        
        # 5. Track token usage
        usage_metadata = getattr(response, 'usage_metadata', None)
        if usage_metadata:
            input_tokens = getattr(usage_metadata, 'prompt_token_count', token_info['input_tokens'])
            output_tokens = getattr(usage_metadata, 'candidates_token_count', 0)
            cached_tokens = getattr(usage_metadata, 'cached_content_token_count', 0)
            token_counter.track_api_usage(input_tokens, output_tokens, cached_tokens)
        else:
            # Fallback token tracking
            token_counter.track_api_usage(token_info['input_tokens'], len(analysis_text.split()) * 1.3, 0)
        
        # Clean and parse JSON
        analysis_text = analysis_text.strip()
        if analysis_text.startswith("```json"):
            analysis_text = analysis_text[7:]
        if analysis_text.endswith("```"):
            analysis_text = analysis_text[:-3]
        
        import json
        result = json.loads(analysis_text)
        
        # 6. Cache the result for future use
        cache_system.store_cached_result(cache_key, result, ttl_hours=2)
        print(f"💾 Cached analysis result for document {doc_id}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error generating analysis: {e}")
        # Track failed request
        token_counter.track_api_usage(token_info.get('input_tokens', 0), 0, 0)
        
        return {
            "summary": "Analysis failed",
            "clauseCategories": [],
            "riskScore": 50,
            "overallRisk": "medium",
            "keyTerms": [],
            "jurisdiction": "unknown",
            "documentType": "unknown"
        }

def generate_comparison_analysis(document_analyses):
    """Generate detailed comparison between documents with caching and token counting."""
    
    # Get instances
    cache_system = get_cache_system()
    token_counter = get_token_counter()
    
    # 1. Generate cache key based on document IDs and analysis content
    doc_ids = [doc.get('id', 'unknown') for doc in document_analyses]
    cache_key = cache_system.generate_cache_key("comparison", *doc_ids, len(document_analyses))
    
    # 2. Check for cached comparison
    cached_result = cache_system.get_cached_result(cache_key)
    if cached_result:
        print(f"✅ Using cached comparison for documents: {doc_ids}")
        return cached_result
    
    from google import genai
    from google.genai import types
    
    _API_KEY = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=_API_KEY)
    
    # Prepare detailed documents for comparison
    docs_detail = []
    for i, doc in enumerate(document_analyses):
        try:
            analysis = doc.get('analysis', {})
            if not analysis:
                print(f"⚠️ Warning: Document {i+1} has no analysis data")
                continue
            
            # Extract detailed clause information with safety checks
            clause_details = []
            for category in analysis.get('clauseCategories', []):
                if not isinstance(category, dict):
                    continue
                category_text = f"Category: {category.get('category', 'Unknown')}\n"
                category_text += f"Risk Level: {category.get('riskLevel', 'medium')}\n"
                category_text += "Clauses:\n"
                clauses = category.get('clauses', [])
                if isinstance(clauses, list):
                    for clause in clauses:
                        if clause:  # Ensure clause is not None or empty
                            category_text += f"- {clause}\n"
                clause_details.append(category_text)
            
            # Extract key terms and conditions safely
            key_terms = analysis.get('keyTerms', [])
            if isinstance(key_terms, list):
                key_terms_text = "Key Terms: " + ", ".join(str(term) for term in key_terms if term)
            else:
                key_terms_text = "Key Terms: None specified"
            
            # Build document detail string with safe defaults
            doc_name = doc.get('name', f'Document {i+1}')
            summary = analysis.get('summary', 'No summary available')
            risk_score = analysis.get('riskScore', 50)
            overall_risk = analysis.get('overallRisk', 'medium')
            jurisdiction = analysis.get('jurisdiction', 'unknown')
            
            docs_detail.append(f"""
            Document {i+1} ({doc_name}):
            Summary: {summary}
            Risk Score: {risk_score}/100
            Overall Risk: {overall_risk}
            Jurisdiction: {jurisdiction}
            
            {key_terms_text}
            
            Detailed Clause Analysis:
            {chr(10).join(clause_details) if clause_details else 'No clause analysis available'}
            """)
            
        except Exception as e:
            print(f"⚠️ Error processing document {i+1} for comparison: {e}")
            # Add a fallback document description
            doc_name = doc.get('name', f'Document {i+1}')
            docs_detail.append(f"""
            Document {i+1} ({doc_name}):
            Summary: Error processing document analysis
            Risk Score: 50/100
            Overall Risk: medium
            Note: This document could not be fully processed for comparison
            """)
            continue
        
        docs_detail.append(f"""
        Document {i+1} ({doc['name']}):
        Summary: {analysis.get('summary', '')}
        Risk Score: {analysis.get('riskScore', 50)}/100
        Overall Risk: {analysis.get('overallRisk', 'medium')}
        Jurisdiction: {analysis.get('jurisdiction', 'unknown')}
        
        {key_terms_text}
        
        Detailed Clause Analysis:
        {chr(10).join(clause_details)}
        """)
    
    comparison_prompt = f"""
    Compare these legal documents in detail and identify specific differences, risks, and recommendations:
    
    {chr(10).join(docs_detail)}
    
    Provide a comprehensive comparison in JSON format. Focus on practical differences that matter for legal risk and business outcomes:
    
    {{
        "clauseDifferences": [
            {{
                "category": "Payment Terms",
                "document1": ["specific clauses from document 1"],
                "document2": ["specific clauses from document 2"],
                "differences": ["detailed explanation of key differences"],
                "riskComparison": {{
                    "doc1Risk": "low|medium|high",
                    "doc2Risk": "low|medium|high", 
                    "betterDocument": "Document 1|Document 2",
                    "reasoning": "detailed explanation why one is better"
                }}
            }}
        ],
        "overallComparison": {{
            "doc1Score": 75,
            "doc2Score": 68,
            "betterDocument": "Document 1|Document 2|Both are similar",
            "riskSummary": "comprehensive summary of overall risk comparison with specific recommendations"
        }},
        "missingClauses": [
            {{
                "category": "Termination Clauses",
                "missingFrom": "Document 1|Document 2", 
                "importance": "high|medium|low",
                "recommendation": "specific recommendation for addressing this gap",
                "riskImpact": "explanation of risk if not addressed"
            }}
        ],
        "recommendations": [
            "Specific actionable recommendation 1",
            "Specific actionable recommendation 2"
        ]
    }}
    
    Ensure clauseDifferences contains at least 3-5 meaningful comparisons across different clause categories.
    Make differences and recommendations specific and actionable.
    """
    
    # 3. Count tokens before API call
    token_info = token_counter.count_tokens_before_request(comparison_prompt)
    print(f"📊 Comparison request - Estimated tokens: {token_info['input_tokens']}, Cost: ${token_info['estimated_cost_usd']:.4f}")
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=comparison_prompt,
            config=types.GenerateContentConfig(
                temperature=0.2
            )
        )
        comparison_text = response.text if hasattr(response, 'text') else "{}"
        
        # 4. Track token usage
        usage_metadata = getattr(response, 'usage_metadata', None)
        if usage_metadata:
            input_tokens = getattr(usage_metadata, 'prompt_token_count', None) or token_info.get('input_tokens', 0)
            output_tokens = getattr(usage_metadata, 'candidates_token_count', None) or 0
            cached_tokens = getattr(usage_metadata, 'cached_content_token_count', None) or 0
            token_counter.track_api_usage(input_tokens, output_tokens, cached_tokens)
        else:
            # Fallback token tracking
            fallback_output = max(len(comparison_text.split()) * 1.3, 0) if comparison_text else 0
            token_counter.track_api_usage(token_info.get('input_tokens', 0), int(fallback_output), 0)
        
        # Clean and parse JSON
        comparison_text = comparison_text.strip()
        if comparison_text.startswith("```json"):
            comparison_text = comparison_text[7:]
        if comparison_text.endswith("```"):
            comparison_text = comparison_text[:-3]
        
        import json
        result = json.loads(comparison_text)
        
        # 5. Cache the comparison result
        cache_system.store_cached_result(cache_key, result, ttl_hours=1)
        print(f"💾 Cached comparison result for documents: {doc_ids}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error generating comparison: {e}")
        # Track failed request
        token_counter.track_api_usage(token_info.get('input_tokens', 0), 0, 0)
        
        return {
            "clauseDifferences": [],
            "overallComparison": {
                "doc1Score": 50,
                "doc2Score": 50,
                "betterDocument": "Both documents have similar risk profiles",
                "riskSummary": "Comparison analysis failed"
            },
            "missingClauses": [],
            "recommendations": ["Manual review recommended due to analysis error"]
        }

@app.post("/api/documents/comparison/export-pdf")
def export_comparison_pdf(comparison_data: dict = Body(...), user=Depends(verify_firebase_token)):
    """Export document comparison as PDF."""
    try:
        from fpdf import FPDF
        import io
        from fastapi.responses import StreamingResponse
        import re
        
        def clean_text_for_pdf(text):
            if not text:
                return ""
            text = text.replace('\u2022', '-')
            text = text.replace('\u2013', '-')
            text = text.replace('\u2014', '--')
            text = text.replace('\u2018', "'")
            text = text.replace('\u2019', "'")
            text = text.replace('\u201c', '"')
            text = text.replace('\u201d', '"')
            text = text.replace('\u2026', '...')
            text = re.sub(r'[^\x00-\x7F]+', '', text)
            return text
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Title
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Document Comparison Report", ln=True, align="C")
        pdf.ln(5)
        
        # Documents being compared
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Documents Compared:", ln=True)
        pdf.set_font("Arial", size=10)
        for i, doc in enumerate(comparison_data.get('documents', []), 1):
            doc_name = clean_text_for_pdf(doc.get('name', f'Document {i}'))
            pdf.cell(0, 6, f"{i}. {doc_name}", ln=True)
        pdf.ln(5)
        
        # Overall comparison
        overall = comparison_data.get('comparison', {}).get('overallComparison', {})
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Overall Comparison:", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 6, f"Better Document: {clean_text_for_pdf(overall.get('betterDocument', 'N/A'))}", ln=True)
        
        risk_summary = clean_text_for_pdf(overall.get('riskSummary', ''))
        if len(risk_summary) > 80:
            risk_summary = risk_summary[:77] + "..."
        pdf.cell(0, 6, f"Summary: {risk_summary}", ln=True)
        pdf.ln(5)
        
        # Clause differences (limit to first 5)
        clause_diffs = comparison_data.get('comparison', {}).get('clauseDifferences', [])[:5]
        if clause_diffs:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "Key Clause Differences:", ln=True)
            pdf.set_font("Arial", size=9)
            
            for diff in clause_diffs:
                category = clean_text_for_pdf(diff.get('category', 'Unknown'))
                pdf.set_font("Arial", "B", 9)
                pdf.cell(0, 6, f"Category: {category}", ln=True)
                
                # Show differences
                differences = diff.get('differences', [])
                if differences:
                    pdf.set_font("Arial", size=8)
                    pdf.cell(0, 5, "Key Differences:", ln=True)
                    for i, difference in enumerate(differences[:3], 1):  # Limit to 3 differences per category
                        diff_text = clean_text_for_pdf(difference)
                        if len(diff_text) > 80:
                            diff_text = diff_text[:77] + "..."
                        pdf.cell(0, 4, f"  {i}. {diff_text}", ln=True)
                
                # Show document clauses
                doc1_clauses = diff.get('document1', [])
                doc2_clauses = diff.get('document2', [])
                if doc1_clauses or doc2_clauses:
                    if doc1_clauses:
                        pdf.set_font("Arial", "B", 8)
                        pdf.cell(0, 4, "Document 1 Clauses:", ln=True)
                        pdf.set_font("Arial", size=7)
                        for clause in doc1_clauses[:2]:  # Limit to 2 clauses
                            clause_text = clean_text_for_pdf(clause)
                            if len(clause_text) > 70:
                                clause_text = clause_text[:67] + "..."
                            pdf.cell(0, 4, f"  - {clause_text}", ln=True)
                    
                    if doc2_clauses:
                        pdf.set_font("Arial", "B", 8)
                        pdf.cell(0, 4, "Document 2 Clauses:", ln=True)
                        pdf.set_font("Arial", size=7)
                        for clause in doc2_clauses[:2]:  # Limit to 2 clauses
                            clause_text = clean_text_for_pdf(clause)
                            if len(clause_text) > 70:
                                clause_text = clause_text[:67] + "..."
                            pdf.cell(0, 4, f"  - {clause_text}", ln=True)
                
                # Show risk comparison
                risk_comp = diff.get('riskComparison', {})
                if risk_comp:
                    better_doc = clean_text_for_pdf(risk_comp.get('betterDocument', 'N/A'))
                    reasoning = clean_text_for_pdf(risk_comp.get('reasoning', ''))
                    if len(reasoning) > 60:
                        reasoning = reasoning[:57] + "..."
                    pdf.set_font("Arial", size=8)
                    pdf.cell(0, 4, f"Better Option: {better_doc}", ln=True)
                    pdf.cell(0, 4, f"Reasoning: {reasoning}", ln=True)
                
                pdf.ln(3)  # Space between categories
        
        # Missing clauses
        missing = comparison_data.get('comparison', {}).get('missingClauses', [])[:5]
        if missing:
            pdf.ln(3)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "Missing Clauses:", ln=True)
            pdf.set_font("Arial", size=9)
            
            for clause in missing:
                category = clean_text_for_pdf(clause.get('category', 'Unknown'))
                missing_from = clean_text_for_pdf(clause.get('missingFrom', 'Unknown'))
                importance = clause.get('importance', 'medium').upper()
                recommendation = clean_text_for_pdf(clause.get('recommendation', ''))
                risk_impact = clean_text_for_pdf(clause.get('riskImpact', ''))
                
                pdf.set_font("Arial", "B", 9)
                pdf.cell(0, 6, f"- {category}", ln=True)
                pdf.set_font("Arial", size=8)
                pdf.cell(0, 4, f"  Missing from: {missing_from} (Importance: {importance})", ln=True)
                
                if recommendation and len(recommendation) > 10:
                    if len(recommendation) > 70:
                        recommendation = recommendation[:67] + "..."
                    pdf.cell(0, 4, f"  Recommendation: {recommendation}", ln=True)
                
                if risk_impact and len(risk_impact) > 10:
                    if len(risk_impact) > 70:
                        risk_impact = risk_impact[:67] + "..."
                    pdf.cell(0, 4, f"  Risk Impact: {risk_impact}", ln=True)
                
                pdf.ln(1)
        
        # Recommendations
        recommendations = comparison_data.get('comparison', {}).get('recommendations', [])[:5]
        if recommendations:
            pdf.ln(3)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "Recommendations:", ln=True)
            pdf.set_font("Arial", size=9)
            
            for i, rec in enumerate(recommendations, 1):
                rec_clean = clean_text_for_pdf(rec)
                if len(rec_clean) > 70:
                    rec_clean = rec_clean[:67] + "..."
                pdf.cell(0, 6, f"{i}. {rec_clean}", ln=True)
        
        # Footer
        pdf.ln(10)
        pdf.set_font("Arial", "I", 8)
        pdf.cell(0, 6, "This comparison is generated by AI and should not replace professional legal advice.", ln=True)
        
        # Return PDF
        pdf_output = pdf.output(dest='S')
        if isinstance(pdf_output, str):
            pdf_output = pdf_output.encode('latin1')
        
        return StreamingResponse(
            io.BytesIO(pdf_output),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=document_comparison.pdf"}
        )
        
    except Exception as e:
        print(f"Error generating comparison PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate comparison PDF: {str(e)}")

# Add this import for title generation
import re

@app.post("/api/chat/session")
def create_chat_session(data: dict = Body(...), user=Depends(verify_firebase_token)):
    """Create a new chat session for a document."""
    document_id = data.get("documentId")
    session_type = data.get("type", "chat")  # Default to "chat", can be "comparison"
    document_ids = data.get("document_ids", [])  # For comparison sessions
    
    session = {
        "userId": user["uid"],
        "documentId": document_id,
        "messages": [],
        "createdAt": firestore.SERVER_TIMESTAMP,
        "title": data.get("title") or "New Chat",
        "type": session_type
    }
    
    # For comparison sessions, store document IDs
    if session_type == "comparison" and document_ids:
        session["document_ids"] = document_ids
        session["documentId"] = None  # No single document for comparison
    
    session_id = add_qa_session(db, session)  # Pass db
    
    # IMPORTANT: Update the session to include its own ID as a field
    update_qa_session_field(db, session_id, "session_id", session_id)  # Pass db
    
    return {"session_id": session_id}

@app.post("/api/chat/session/new")
def create_new_chat_session(data: dict = Body(...), user=Depends(verify_firebase_token)):
    """Create a new chat session (alternative endpoint)."""
    return create_chat_session(data, user)

def generate_title_from_message(message_text: str) -> str:
    """Generate a title from the first user message."""
    # Clean the message
    clean_text = re.sub(r'[^\w\s]', '', message_text)
    words = clean_text.split()
    
    # Take first 4-6 words, max 40 characters
    if len(words) <= 4:
        title = ' '.join(words)
    else:
        title = ' '.join(words[:4])
    
    # Truncate if too long
    if len(title) > 40:
        title = title[:37] + "..."
    
    return title.strip() or "New Chat"

@app.post("/api/chat/session/{session_id}/message")
def add_message_to_session(session_id: str, data: dict = Body(...), user=Depends(verify_firebase_token)):
    """Add a message to a chat session and get AI response."""
    session = get_qa_session_by_id(db, session_id)  # Pass db
    if not session or session.get("userId") != user["uid"]:
        raise HTTPException(status_code=404, detail="Session not found")
    
    document_id = session["documentId"]
    
    # Check if this is the first message to generate title
    is_first_message = len(session.get("messages", [])) == 0
    
    # Create user message with current timestamp
    current_time = datetime.utcnow().isoformat()
    user_message = {
        "role": "user", 
        "text": data["text"], 
        "timestamp": current_time
    }
    
    # Generate AI response (same logic as before)
    chunks = get_chunks_by_doc_id(db, document_id)  # Pass db
    chunk_texts = [c["text"] for c in chunks]
    chunk_embs = [c["embedding"] for c in chunks]
    query_emb = embed_text(data["text"])
    sim_scores = [cosine_similarity(query_emb, emb) for emb in chunk_embs]
    pool_size = min(50, len(chunks))
    top_pool_idxs = sorted(range(len(sim_scores)), key=lambda i: sim_scores[i], reverse=True)[:pool_size]
    pool_embs = [chunk_embs[i] for i in top_pool_idxs]
    pool_texts = [chunk_texts[i] for i in top_pool_idxs]
    K = min(8, len(pool_embs))
    selected_idxs = mmr(query_emb, pool_embs, K=K, lambda_=0.7)
    selected_texts = [pool_texts[i] for i in selected_idxs]
    
    from google import genai
    from google.genai import types
    
    _API_KEY = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=_API_KEY)
    
    context = "\n".join(selected_texts)
    prompt = f"Context: {context}\nQuestion: {data['text']}\nAnswer in markdown format in ≤ 120 words. Use appropriate markdown formatting like **bold**, *italic*, `code`, bullet points, etc. If uncertain, respond 'I don't know — please consult a lawyer' and show the top 2 source snippets used."
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1
        )
    )
    
    ai_message = {
        "role": "ai", 
        "text": response.text if hasattr(response, 'text') else "No answer.", 
        "timestamp": current_time
    }
    
    # Update session with new messages
    update_qa_session_messages(db, session_id, [user_message, ai_message])  # Pass db
    
    # Generate and update title if this is the first message
    updated_session = session  # Default
    if is_first_message:
        new_title = generate_title_from_message(data["text"])
        update_qa_session_field(db, session_id, "title", new_title)  # Pass db
        # Get updated session
        updated_session = get_qa_session_by_id(db, session_id)  # Pass db
    
    return {
        "messages": [user_message, ai_message],
        "session": updated_session  # Include updated session info
    }

from fastapi.responses import StreamingResponse
import json

@app.post("/api/chat/session/{session_id}/message/stream")
def stream_message_response(session_id: str, data: dict = Body(...), user=Depends(verify_firebase_token)):
    """Stream AI response for a chat message."""
    session = get_qa_session_by_id(db, session_id)
    if not session or session.get("userId") != user["uid"]:
        raise HTTPException(status_code=404, detail="Session not found")
    
    document_id = session["documentId"]
    
    # Check if this is the first message to generate title
    is_first_message = len(session.get("messages", [])) == 0
    
    # Create user message with current timestamp
    current_time = datetime.utcnow().isoformat()
    user_message = {
        "role": "user", 
        "text": data["text"], 
        "timestamp": current_time
    }
    
    def generate_stream():
        try:
            # Generate AI response
            chunks = get_chunks_by_doc_id(db, document_id)
            chunk_texts = [c["text"] for c in chunks]
            chunk_embs = [c["embedding"] for c in chunks]
            query_emb = embed_text(data["text"])
            sim_scores = [cosine_similarity(query_emb, emb) for emb in chunk_embs]
            pool_size = min(50, len(chunks))
            top_pool_idxs = sorted(range(len(sim_scores)), key=lambda i: sim_scores[i], reverse=True)[:pool_size]
            pool_embs = [chunk_embs[i] for i in top_pool_idxs]
            pool_texts = [chunk_texts[i] for i in top_pool_idxs]
            K = min(8, len(pool_embs))
            selected_idxs = mmr(query_emb, pool_embs, K=K, lambda_=0.7)
            selected_texts = [pool_texts[i] for i in selected_idxs]
            
            from google import genai
            from google.genai import types
            
            _API_KEY = os.getenv("GEMINI_API_KEY")
            client = genai.Client(api_key=_API_KEY)
            
            context = "\n".join(selected_texts)
            prompt = f"Context: {context}\nQuestion: {data['text']}\nAnswer in markdown format in ≤ 120 words. Use appropriate markdown formatting like **bold**, *italic*, `code`, bullet points, etc. If uncertain, respond 'I don't know — please consult a lawyer' and show the top 2 source snippets used."
            
            # Send user message first
            yield f"data: {json.dumps({'type': 'user_message', 'message': user_message})}\n\n"
            
            # Stream AI response
            response_stream = client.models.generate_content_stream(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1
                )
            )
            
            accumulated_text = ""
            for chunk in response_stream:
                if chunk.text:
                    accumulated_text += chunk.text
                    yield f"data: {json.dumps({'type': 'ai_chunk', 'chunk': chunk.text, 'accumulated': accumulated_text})}\n\n"
            
            # Create final AI message
            ai_message = {
                "role": "ai",
                "text": accumulated_text,
                "timestamp": current_time
            }
            
            # Update session with new messages
            update_qa_session_messages(db, session_id, [user_message, ai_message])
            
            # Generate and update title if this is the first message
            updated_session = session
            if is_first_message:
                new_title = generate_title_from_message(data["text"])
                update_qa_session_field(db, session_id, "title", new_title)
                updated_session = get_qa_session_by_id(db, session_id)
            
            # Send completion message
            yield f"data: {json.dumps({'type': 'complete', 'ai_message': ai_message, 'session': updated_session})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )

@app.get("/api/documents/{doc_id}/status")
def get_document_status(doc_id: str, user=Depends(verify_firebase_token)):
    """Get the processing status of a document."""
    try:
        # First check the document status in Firestore
        doc_ref = db.collection(os.getenv("FIRESTORE_DOCUMENTS_COLLECTION", "documents")).document(doc_id)
        doc_snapshot = doc_ref.get()
        
        if not doc_snapshot.exists:
            return {"status": "error", "message": "Document not found"}
        
        doc_data = doc_snapshot.to_dict()
        firestore_status = doc_data.get('status', 'unknown')
        
        print(f"Document {doc_id} Firestore status: {firestore_status}")
        
        # Only return ready if document is fully processed AND has chunks
        if firestore_status in ['processed', 'processed_with_summary_error']:
            # Double-check that chunks exist
            chunks = get_chunks_by_doc_id(db, doc_id)
            if chunks and len(chunks) > 0:
                print(f"Document {doc_id} is ready with {len(chunks)} chunks")
                return {"status": "ready", "message": "Document ready for analysis"}
            else:
                print(f"Document {doc_id} marked as processed but no chunks found")
                return {"status": "processing", "message": "Finalizing document processing"}
        elif firestore_status == 'failed':
            return {"status": "error", "message": "Document processing failed"}
        elif firestore_status in ['processing', 'uploaded']:
            return {"status": "processing", "message": "Document is being processed"}
        else:
            print(f"Document {doc_id} has unknown status: {firestore_status}")
            return {"status": "processing", "message": f"Document status: {firestore_status}"}
    
    except Exception as e:
        print(f"Error checking document status for {doc_id}: {e}")
        return {"status": "error", "message": "Error checking document status"}

@app.get("/api/chat/sessions")
def list_chat_sessions(user=Depends(verify_firebase_token)):
    """List all chat sessions for the user."""
    sessions = get_qa_sessions_by_user(db, user["uid"])  # Pass db
    return {"sessions": sessions}

@app.get("/api/chat/session/{session_id}")
def get_chat_session(session_id: str, user=Depends(verify_firebase_token)):
    session = get_qa_session_by_id(db, session_id)  # Pass db
    if not session or session.get("userId") != user["uid"]:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

# Phase 3.2: Cache and Token Management Endpoints

@app.get("/api/admin/cache/stats")
def get_cache_statistics(user=Depends(verify_firebase_token)):
    """Get cache performance statistics and token usage analytics."""
    try:
        # Get instances
        cache_system = get_cache_system()
        token_counter = get_token_counter()
        
        # Get cache statistics
        cache_stats = cache_system.get_cache_statistics()
        
        # Get token counter statistics
        token_stats = token_counter.get_session_stats()
        
        # Get detailed analytics
        analytics = {
            "cache_performance": {
                "total_requests": cache_stats["total_requests"],
                "cache_hits": cache_stats["cache_hits"],
                "cache_misses": cache_stats["cache_misses"],
                "hit_rate": f"{cache_stats['hit_rate']:.2f}%",
                "total_caches_created": cache_stats["gemini_caches_created"],
                "active_caches": len(cache_stats["active_caches"]),
                "memory_cache_entries": cache_stats["memory_cache_entries"]
            },
            "token_usage": {
                "total_input_tokens": token_stats["total_input_tokens"],
                "total_output_tokens": token_stats["total_output_tokens"],
                "total_cached_tokens": token_stats["total_cached_tokens"],
                "total_cost_usd": token_stats["total_cost_usd"],
                "total_api_calls": token_stats["total_api_calls"],
                "average_tokens_per_call": token_stats["average_tokens_per_call"],
                "cache_savings_usd": token_stats["cache_savings_usd"],
                "session_start_time": token_stats["session_start_time"].isoformat() if token_stats["session_start_time"] else None
            },
            "cost_savings": {
                "estimated_savings_without_cache": f"${(token_stats['total_input_tokens'] * 0.000025):.4f}",
                "actual_cost_with_cache": f"${token_stats['total_cost_usd']:.4f}",
                "savings_percentage": f"{(token_stats['cache_savings_usd'] / max(token_stats['total_cost_usd'], 0.0001)) * 100:.1f}%"
            }
        }
        
        print(f"📊 Cache and token statistics requested by user {user['uid']}")
        return analytics
        
    except Exception as e:
        print(f"❌ Error getting cache statistics: {e}")
        return {
            "error": "Failed to retrieve statistics",
            "cache_performance": {},
            "token_usage": {},
            "cost_savings": {}
        }

@app.post("/api/admin/cache/cleanup")
def cleanup_caches(user=Depends(verify_firebase_token)):
    """Manually trigger cache cleanup to remove expired entries."""
    try:
        # Get instance
        cache_system = get_cache_system()
        
        # Clean up memory cache
        memory_cleanup_count = cache_system.cleanup_expired_cache()
        
        # Clean up Gemini caches (this will mark expired ones for deletion)
        gemini_cleanup_count = cache_system.cleanup_gemini_caches()
        
        cleanup_stats = {
            "memory_cache_cleaned": memory_cleanup_count,
            "gemini_caches_cleaned": gemini_cleanup_count,
            "cleanup_timestamp": datetime.utcnow().isoformat(),
            "status": "success"
        }
        
        print(f"🧹 Cache cleanup completed: {memory_cleanup_count} memory entries, {gemini_cleanup_count} Gemini caches")
        return cleanup_stats
        
    except Exception as e:
        print(f"❌ Error during cache cleanup: {e}")
        return {
            "error": "Cache cleanup failed",
            "cleanup_timestamp": datetime.utcnow().isoformat(),
            "status": "error"
        }

@app.post("/api/admin/cache/reset")
def reset_cache_system(user=Depends(verify_firebase_token)):
    """Reset the entire cache system and token counters (admin only)."""
    try:
        # Get instances
        cache_system = get_cache_system()
        token_counter = get_token_counter()
        
        # Clear all memory cache
        cache_system.memory_cache.clear()
        
        # Reset token counter
        token_counter.reset_session_stats()
        
        # Note: We don't automatically delete Gemini caches as they might be expensive to recreate
        # They will expire naturally based on their TTL
        
        reset_info = {
            "memory_cache_cleared": True,
            "token_stats_reset": True,
            "gemini_caches_note": "Gemini caches preserved (will expire based on TTL)",
            "reset_timestamp": datetime.utcnow().isoformat(),
            "status": "success"
        }
        
        print(f"🔄 Cache system reset by user {user['uid']}")
        return reset_info
        
    except Exception as e:
        print(f"❌ Error resetting cache system: {e}")
        return {
            "error": "Cache reset failed",
            "reset_timestamp": datetime.utcnow().isoformat(),
            "status": "error"
        }

@app.get("/api/admin/token/usage")
def get_token_usage_details(user=Depends(verify_firebase_token)):
    """Get detailed token usage information for cost tracking."""
    try:
        # Get instance
        token_counter = get_token_counter()
        
        stats = token_counter.get_session_stats()
        
        # Calculate additional metrics
        session_duration = datetime.utcnow() - stats["session_start_time"] if stats["session_start_time"] else None
        session_hours = session_duration.total_seconds() / 3600 if session_duration else 0
        
        detailed_usage = {
            "session_info": {
                "start_time": stats["session_start_time"].isoformat() if stats["session_start_time"] else None,
                "duration_hours": round(session_hours, 2),
                "total_api_calls": stats["total_api_calls"]
            },
            "token_breakdown": {
                "input_tokens": stats["total_input_tokens"],
                "output_tokens": stats["total_output_tokens"],
                "cached_tokens": stats["total_cached_tokens"],
                "total_tokens": stats["total_input_tokens"] + stats["total_output_tokens"]
            },
            "cost_analysis": {
                "total_cost_usd": stats["total_cost_usd"],
                "input_cost_usd": stats["total_input_tokens"] * 0.000025,
                "output_cost_usd": stats["total_output_tokens"] * 0.000075,
                "cache_savings_usd": stats["cache_savings_usd"],
                "average_cost_per_call": stats["total_cost_usd"] / max(stats["total_api_calls"], 1)
            },
            "efficiency_metrics": {
                "tokens_per_hour": (stats["total_input_tokens"] + stats["total_output_tokens"]) / max(session_hours, 0.1),
                "cache_utilization": f"{(stats['total_cached_tokens'] / max(stats['total_input_tokens'], 1)) * 100:.1f}%",
                "average_tokens_per_call": stats["average_tokens_per_call"]
            }
        }
        
        return detailed_usage
        
    except Exception as e:
        print(f"❌ Error getting token usage details: {e}")
        return {
            "error": "Failed to retrieve token usage details",
            "session_info": {},
            "token_breakdown": {},
            "cost_analysis": {},
            "efficiency_metrics": {}
        }