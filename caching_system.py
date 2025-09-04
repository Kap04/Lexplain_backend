# caching_system.py
import os
import json
import logging
import hashlib
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from google import genai
from google.genai import types

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CachingSystem:
    def __init__(self):
        self._api_key = os.getenv("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        self.client = genai.Client(api_key=self._api_key)
        
        # In-memory cache for development (in production, use Redis or similar)
        self.memory_cache = {}
        self.gemini_caches = {}  # Store Gemini cache IDs
        
        # Cache statistics
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "created": 0,
            "expired": 0
        }
    
    def create_document_cache(self, document_id: str, chunks: List[str], ttl_hours: int = 2) -> Optional[str]:
        """Create a Gemini cache for document chunks."""
        try:
            # Create a hash for the document content
            content_hash = self._hash_content(chunks)
            cache_key = f"doc_{document_id}_{content_hash}"
            
            # Check if we already have a cache for this content
            if cache_key in self.gemini_caches:
                existing_cache = self.gemini_caches[cache_key]
                if existing_cache.get("expires_at", 0) > time.time():
                    logger.info(f"♻️ Reusing existing cache for document {document_id}")
                    return existing_cache["cache_name"]
            
            # Prepare content for caching
            cache_contents = []
            for i, chunk in enumerate(chunks[:20]):  # Limit to first 20 chunks for caching
                cache_contents.append({
                    "role": "user",
                    "parts": [{"text": f"Document chunk {i+1}: {chunk}"}]
                })
            
            # Create the cache
            ttl_seconds = ttl_hours * 3600
            
            cache_response = self.client.caches.create({
                "model": "gemini-2.0-flash-exp",
                "contents": cache_contents,
                "system_instruction": "You are analyzing legal document chunks. Use this cached content for analysis.",
                "ttl": f"{ttl_seconds}s"
            })
            
            cache_name = cache_response.name
            
            # Store cache info
            self.gemini_caches[cache_key] = {
                "cache_name": cache_name,
                "document_id": document_id,
                "created_at": time.time(),
                "expires_at": time.time() + ttl_seconds,
                "chunk_count": len(chunks)
            }
            
            self.cache_stats["created"] += 1
            
            logger.info(f"🗃️ Created document cache: {cache_name}")
            logger.info(f"   Document ID: {document_id}")
            logger.info(f"   Chunks cached: {len(chunks)}")
            logger.info(f"   TTL: {ttl_hours} hours")
            
            return cache_name
            
        except Exception as e:
            logger.error(f"❌ Error creating document cache: {e}")
            return None
    
    def create_analysis_cache(self, analysis_type: str, content: str, ttl_hours: int = 1) -> Optional[str]:
        """Create a cache for analysis results."""
        try:
            content_hash = self._hash_content([content])
            cache_key = f"analysis_{analysis_type}_{content_hash}"
            
            # Check existing cache
            if cache_key in self.gemini_caches:
                existing_cache = self.gemini_caches[cache_key]
                if existing_cache.get("expires_at", 0) > time.time():
                    logger.info(f"♻️ Reusing existing analysis cache for {analysis_type}")
                    return existing_cache["cache_name"]
            
            # Create cache for analysis
            ttl_seconds = ttl_hours * 3600
            
            cache_response = self.client.caches.create({
                "model": "gemini-2.0-flash-exp",
                "contents": [{"role": "user", "parts": [{"text": content}]}],
                "system_instruction": f"You are performing {analysis_type} analysis. Use this cached content.",
                "ttl": f"{ttl_seconds}s"
            })
            
            cache_name = cache_response.name
            
            self.gemini_caches[cache_key] = {
                "cache_name": cache_name,
                "analysis_type": analysis_type,
                "created_at": time.time(),
                "expires_at": time.time() + ttl_seconds
            }
            
            self.cache_stats["created"] += 1
            
            logger.info(f"🧠 Created analysis cache: {cache_name} for {analysis_type}")
            
            return cache_name
            
        except Exception as e:
            logger.error(f"❌ Error creating analysis cache: {e}")
            return None
    
    def get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get result from memory cache."""
        if cache_key in self.memory_cache:
            cache_entry = self.memory_cache[cache_key]
            if cache_entry["expires_at"] > time.time():
                self.cache_stats["hits"] += 1
                logger.info(f"✅ Cache hit for key: {cache_key}")
                return cache_entry["data"]
            else:
                # Cache expired
                del self.memory_cache[cache_key]
                self.cache_stats["expired"] += 1
                logger.info(f"⏰ Cache expired for key: {cache_key}")
        
        self.cache_stats["misses"] += 1
        logger.info(f"❌ Cache miss for key: {cache_key}")
        return None
    
    def store_cached_result(self, cache_key: str, data: Any, ttl_hours: int = 2):
        """Store result in memory cache."""
        expires_at = time.time() + (ttl_hours * 3600)
        self.memory_cache[cache_key] = {
            "data": data,
            "created_at": time.time(),
            "expires_at": expires_at
        }
        logger.info(f"💾 Stored result in cache: {cache_key} (TTL: {ttl_hours}h)")
    
    def generate_cache_key(self, *args) -> str:
        """Generate a cache key from arguments."""
        combined = "_".join(str(arg) for arg in args)
        return hashlib.md5(combined.encode()).hexdigest()
    
    def cleanup_expired_caches(self):
        """Clean up expired caches."""
        current_time = time.time()
        expired_keys = []
        
        # Clean memory cache
        for key, entry in self.memory_cache.items():
            if entry["expires_at"] <= current_time:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.memory_cache[key]
            self.cache_stats["expired"] += 1
        
        # Clean Gemini caches
        expired_gemini = []
        for key, cache_info in self.gemini_caches.items():
            if cache_info["expires_at"] <= current_time:
                expired_gemini.append(key)
        
        for key in expired_gemini:
            try:
                # Delete from Gemini
                cache_name = self.gemini_caches[key]["cache_name"]
                self.client.caches.delete(name=cache_name)
                del self.gemini_caches[key]
                logger.info(f"🗑️ Deleted expired Gemini cache: {cache_name}")
            except Exception as e:
                logger.error(f"Error deleting Gemini cache: {e}")
        
        if expired_keys or expired_gemini:
            logger.info(f"🧹 Cleaned up {len(expired_keys)} memory caches and {len(expired_gemini)} Gemini caches")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_hits": self.cache_stats["hits"],
            "cache_misses": self.cache_stats["misses"],
            "cache_hit_rate_percent": round(hit_rate, 2),
            "caches_created": self.cache_stats["created"],
            "caches_expired": self.cache_stats["expired"],
            "active_memory_caches": len(self.memory_cache),
            "active_gemini_caches": len(self.gemini_caches),
            "total_requests": total_requests
        }
    
    def log_cache_stats(self):
        """Log cache performance statistics."""
        stats = self.get_cache_stats()
        
        logger.info("📈 === CACHE PERFORMANCE STATS ===")
        logger.info(f"Hit Rate: {stats['cache_hit_rate_percent']}%")
        logger.info(f"Hits: {stats['cache_hits']}")
        logger.info(f"Misses: {stats['cache_misses']}")
        logger.info(f"Created: {stats['caches_created']}")
        logger.info(f"Active Memory: {stats['active_memory_caches']}")
        logger.info(f"Active Gemini: {stats['active_gemini_caches']}")
        logger.info("===================================")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics for API endpoints."""
        stats = self.get_cache_stats()
        
        return {
            "total_requests": stats["total_requests"],
            "cache_hits": stats["cache_hits"],
            "cache_misses": stats["cache_misses"],
            "hit_rate": stats["cache_hit_rate_percent"],
            "gemini_caches_created": stats["caches_created"],
            "active_caches": [cache_name for cache_name, info in self.gemini_caches.items() 
                             if info.get("expires_at", 0) > time.time()],
            "memory_cache_entries": len(self.memory_cache)
        }
    
    def cleanup_expired_cache(self) -> int:
        """Clean up expired cache entries and return count of cleaned entries."""
        cleaned_count = 0
        current_time = time.time()
        
        # Clean memory cache
        expired_keys = []
        for key, data in self.memory_cache.items():
            if data.get("expires_at", 0) <= current_time:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.memory_cache[key]
            cleaned_count += 1
        
        logger.info(f"🧹 Cleaned {cleaned_count} expired memory cache entries")
        return cleaned_count
    
    def cleanup_gemini_caches(self) -> int:
        """Mark expired Gemini caches for cleanup and return count."""
        cleaned_count = 0
        current_time = time.time()
        
        expired_caches = []
        for cache_key, cache_info in self.gemini_caches.items():
            if cache_info.get("expires_at", 0) <= current_time:
                expired_caches.append(cache_key)
        
        for cache_key in expired_caches:
            # Note: Gemini caches are automatically cleaned up by Google
            # We just remove our tracking of them
            del self.gemini_caches[cache_key]
            cleaned_count += 1
            self.cache_stats["expired"] += 1
        
        logger.info(f"🧹 Marked {cleaned_count} expired Gemini caches for cleanup")
        return cleaned_count
    
    def _hash_content(self, content_list: List[str]) -> str:
        """Create a hash for content."""
        combined_content = "".join(content_list)
        return hashlib.md5(combined_content.encode()).hexdigest()[:12]

# Global caching instance (lazy initialization)
cache_system = None

def get_cache_system():
    """Get or create the global cache system instance."""
    global cache_system
    if cache_system is None:
        cache_system = CachingSystem()
    return cache_system
