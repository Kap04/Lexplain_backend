# firestore_adapter.py

import os
from google.cloud import firestore
from typing import Dict, Any

# Consistent collection names
COLLECTION_DOCUMENTS = os.getenv("FIRESTORE_DOCUMENTS_COLLECTION", "documents")
COLLECTION_CHUNKS = os.getenv("FIRESTORE_EMBEDDINGS_COLLECTION", "chunks")
COLLECTION_SUMMARIES = os.getenv("FIRESTORE_SUMMARIES_COLLECTION", "summaries")
COLLECTION_QA = os.getenv("FIRESTORE_QA_COLLECTION", "qa_sessions")

# ❌ Remove this line - we'll pass db as parameter instead
# db = firestore.Client()

def add_document_metadata(db: firestore.Client, doc: Dict[str, Any]) -> str:
    ref = db.collection(COLLECTION_DOCUMENTS).add(doc)
    return ref[1].id

def update_document_status(db: firestore.Client, doc_id: str, status: str):
    db.collection(COLLECTION_DOCUMENTS).document(doc_id).update({"status": status})

def add_chunks(db: firestore.Client, chunks: list):
    batch = db.batch()
    for chunk in chunks:
        ref = db.collection(COLLECTION_CHUNKS).document()
        batch.set(ref, chunk)
    batch.commit()

def add_summary(db: firestore.Client, summary: Dict[str, Any]):
    db.collection(COLLECTION_SUMMARIES).add(summary)

def _serialize_firestore_session(data, doc_id):
    """Helper function to serialize Firestore session data"""
    # Convert Firestore timestamp to ISO string if present
    if "createdAt" in data and hasattr(data["createdAt"], "isoformat"):
        data["createdAt"] = data["createdAt"].isoformat()
    
    # Convert message timestamps
    if "messages" in data and isinstance(data["messages"], list):
        for message in data["messages"]:
            if isinstance(message, dict) and "timestamp" in message:
                if hasattr(message["timestamp"], "isoformat"):
                    message["timestamp"] = message["timestamp"].isoformat()
    
    # Ensure consistent session_id field
    data["session_id"] = doc_id
    data["sessionId"] = doc_id  # For backward compatibility
    return data

def add_qa_session(db: firestore.Client, qa: Dict[str, Any]) -> str:
    """Add a new QA session and return its ID, setting session_id at creation."""
    temp_ref = db.collection(COLLECTION_QA).document()
    session_id = temp_ref.id
    qa = dict(qa)
    qa["session_id"] = session_id
    qa["sessionId"] = session_id
    temp_ref.set(qa)
    return session_id

def get_qa_sessions_by_user(db: firestore.Client, user_id: str):
    """Get all QA sessions for a user, ordered by creation date."""
    sessions = []
    try:
        docs = db.collection(COLLECTION_QA)\
                .where("userId", "==", user_id)\
                .order_by("createdAt", direction=firestore.Query.DESCENDING)\
                .stream()
        
        for doc in docs:
            data = doc.to_dict()
            if data:  # Ensure data exists
                sessions.append(_serialize_firestore_session(data, doc.id))
    except Exception as e:
        print(f"Error fetching sessions for user {user_id}: {e}")
    
    return sessions

def get_qa_session_by_id(db: firestore.Client, session_id: str):
    """Get a QA session by its ID."""
    try:
        doc = db.collection(COLLECTION_QA).document(session_id).get()
        if doc.exists:
            data = doc.to_dict()
            if data:
                return _serialize_firestore_session(data, doc.id)
    except Exception as e:
        print(f"Error fetching session {session_id}: {e}")
    
    return None

def update_qa_session_messages(db: firestore.Client, session_id: str, new_messages: list):
    """Append new messages to a QA session."""
    try:
        # Clean messages before storing
        def clean_message(msg):
            msg = dict(msg)
            if "timestamp" in msg and hasattr(msg["timestamp"], "isoformat"):
                msg["timestamp"] = msg["timestamp"].isoformat()
            elif "timestamp" not in msg:
                from datetime import datetime
                msg["timestamp"] = datetime.utcnow().isoformat()
            return msg
        
        cleaned_messages = [clean_message(m) for m in new_messages]
        
        db.collection(COLLECTION_QA).document(session_id).update({
            "messages": firestore.ArrayUnion(cleaned_messages)
        })
    except Exception as e:
        print(f"Error updating messages for session {session_id}: {e}")
        raise e

def update_qa_session_field(db: firestore.Client, session_id: str, field_name: str, field_value):
    """Update a specific field in a QA session."""
    try:
        db.collection(COLLECTION_QA).document(session_id).update({
            field_name: field_value
        })
    except Exception as e:
        print(f"Error updating field {field_name} for session {session_id}: {e}")
        raise e

def delete_qa_session(db: firestore.Client, session_id: str):
    """Delete a QA session."""
    try:
        db.collection(COLLECTION_QA).document(session_id).delete()
    except Exception as e:
        print(f"Error deleting session {session_id}: {e}")
        raise e

def get_summary_by_doc_id(db: firestore.Client, doc_id: str):
    """Get document summary by document ID."""
    try:
        docs = db.collection(COLLECTION_SUMMARIES).where("documentId", "==", doc_id).stream()
        for doc in docs:
            return doc.to_dict()
    except Exception as e:
        print(f"Error fetching summary for document {doc_id}: {e}")
    
    return None

def get_chunks_by_doc_id(db: firestore.Client, doc_id: str):
    """
    Retrieve all chunks for a given document ID.
    Each chunk contains both the original text and the embedding.
    """
    chunks = []
    try:
        docs = db.collection(COLLECTION_CHUNKS).where("documentId", "==", doc_id).stream()

        for doc in docs:
            data = doc.to_dict()
            if data:
                chunks.append({
                    "chunkId": doc.id,
                    "text": data.get("text"),           # human-readable chunk text
                    "embedding": data.get("embedding"), # vector for similarity search
                    "metadata": data.get("metadata", {}) # optional, e.g., page number
                })
    except Exception as e:
        print(f"Error fetching chunks for document {doc_id}: {e}")
    
    return chunks