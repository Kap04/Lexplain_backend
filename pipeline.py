#pipeline.py
# --- PDF Text Extraction (PyMuPDF) ---
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract readable text from a PDF using PyMuPDF."""
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        page_text = page.get_text()
        if page_text:
            text += page_text + "\n"
    return text
from typing import List, Dict, Any
import inspect
import google.genai as genai
from google.genai.types import GenerateContentConfig
import os
import time
import random
import threading
from collections import deque
from datetime import datetime, date, timedelta
try:
    from zoneinfo import ZoneInfo
    _HAS_ZONEINFO = True
except Exception:
    _HAS_ZONEINFO = False


# Add this debug function to your pipeline.py

def debug_api_connection():
    """Test API connection and quota status"""
    try:
        print(f"Testing Gemini API connection...")
        print(f"API Key present: {'GEMINI_API_KEY' in os.environ}")
        print(f"Rate limits: RPM={_RPM}, TPM={_TPM}, RPD={_RPD}")
        print(f"Quota available: {_RATE_LIMITER.is_quota_available()}")
        
        # Test a minimal embedding request
        test_result = embed_text("Hello world")
        print(f"Test embedding successful: {len(test_result)} dimensions")
        return True
    except Exception as e:
        print(f"API connection test failed: {e}")
        return False

# Call this at the start of your main processing function
def process_document_with_debug(*args, **kwargs):
    print("=== Starting Railway Debug ===")
    debug_api_connection()
    print("=== End Railway Debug ===")
    # Continue with your normal processing...

# --- Chunking ---
def chunk_text(pages: List[Dict[str, Any]], chunk_size=500, overlap=50) -> List[Dict[str, Any]]:
    chunks = []
    for page in pages:
        words = page["text"].split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i+chunk_size]
            chunk_text = ' '.join(chunk_words)
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "startPage": page["page"],
                    "endPage": page["page"],
                    "tokens": len(chunk_words)
                })
    return chunks

# --- Embedding (Gemini) ---
_GEMINI_MODEL = "gemini-embedding-001"
_API_KEY = os.getenv("GEMINI_API_KEY")
if not _API_KEY:
    raise RuntimeError("Google AI Studio API key not set in GEMINI_API_KEY")

# Initialize modern client
client = genai.Client(api_key=_API_KEY)

# --- Rate limiter with better quota handling ---
_RPM = int(os.getenv("GEMINI_EMBEDDING_RPM", 5))  # Reduced from 100
_TPM = int(os.getenv("GEMINI_EMBEDDING_TPM", 500))  # Reduced from 30000
_RPD = int(os.getenv("GEMINI_EMBEDDING_RPD", 100))   # Reduced from 1000


def _pt_date_now():
    """Get current date in a timezone-safe way."""
    try:
        if _HAS_ZONEINFO:
            return datetime.now(ZoneInfo("America/Los_Angeles")).date()
    except Exception:
        # Fallback to UTC if timezone not available (Windows)
        pass
    return (datetime.utcnow() - timedelta(hours=8)).date()


def _estimate_tokens_for_text(text: str) -> int:
    if not text:
        return 1
    return max(1, int(len(text) / 2))


# Updated RateLimiter class with Railway-specific fixes

class RateLimiter:
    def __init__(self, rpm=_RPM, tpm=_TPM, rpd=_RPD):
        self.rpm = rpm
        self.tpm = tpm
        self.rpd = rpd
        self.req_timestamps = deque()
        self.token_timestamps = deque()
        self.daily_count = 0
        self.last_reset_date = _pt_date_now()
        self.lock = threading.Lock()
        self.quota_exhausted = False
        self.quota_reset_time = None
        # Add consecutive failures tracking
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3

    def _reset_daily_if_needed(self):
        today = _pt_date_now()
        if today != self.last_reset_date:
            self.daily_count = 0
            self.last_reset_date = today
            # Reset quota exhaustion flag daily
            self.quota_exhausted = False
            self.quota_reset_time = None
            self.consecutive_failures = 0  # Reset failure count

    def mark_quota_exhausted(self, reset_time_hours=24):
        """Mark quota as exhausted with estimated reset time"""
        with self.lock:
            self.quota_exhausted = True
            self.quota_reset_time = time.time() + (reset_time_hours * 3600)
            print(f"Quota marked as exhausted. Estimated reset in {reset_time_hours} hours.")

    def mark_failure(self):
        """Track consecutive failures to detect quota issues early"""
        with self.lock:
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.max_consecutive_failures:
                print(f"Too many consecutive failures ({self.consecutive_failures}), marking quota as exhausted")
                self.mark_quota_exhausted(1)  # Shorter reset time for failure-based exhaustion

    def mark_success(self):
        """Reset failure counter on successful request"""
        with self.lock:
            self.consecutive_failures = 0

    def is_quota_available(self):
        """Check if quota might be available again"""
        with self.lock:
            if not self.quota_exhausted:
                return True
            if self.quota_reset_time and time.time() > self.quota_reset_time:
                print("Quota reset time passed, attempting to clear exhaustion flag")
                self.quota_exhausted = False
                self.quota_reset_time = None
                self.consecutive_failures = 0
                return True
            return False

    def acquire(self, estimated_tokens: int):
        if not self.is_quota_available():
            reset_in = (self.quota_reset_time - time.time()) / 3600 if self.quota_reset_time else 24
            raise RuntimeError(f"Quota exhausted. Try again in ~{reset_in:.1f} hours.")
        
        while True:
            with self.lock:
                now = time.time()
                self._reset_daily_if_needed()
                if self.daily_count >= self.rpd:
                    raise RuntimeError("Daily requests quota reached for Gemini embeddings")
                
                # Clean old entries (older than 60s)
                while self.req_timestamps and now - self.req_timestamps[0] > 60:
                    self.req_timestamps.popleft()
                while self.token_timestamps and now - self.token_timestamps[0][0] > 60:
                    self.token_timestamps.popleft()
                
                current_rpm = len(self.req_timestamps)
                current_tpm = sum(t for ts, t in self.token_timestamps)
                
                # More conservative wait calculation
                wait_rpm = 0.0
                if current_rpm >= self.rpm:
                    oldest = self.req_timestamps[0]
                    wait_rpm = 60 - (now - oldest) + 1  # Add 1 second buffer
                
                wait_tpm = 0.0
                if current_tpm + estimated_tokens > self.tpm:
                    needed = (current_tpm + estimated_tokens) - self.tpm
                    acc = 0
                    wait_tpm = 0.0
                    for ts, t in self.token_timestamps:
                        acc += t
                        if acc >= needed:
                            wait_tpm = 60 - (now - ts) + 1  # Add 1 second buffer
                            break
                    if wait_tpm < 0:
                        wait_tpm = 0.0
                
                wait = max(wait_rpm, wait_tpm)
                if wait <= 0:
                    self.req_timestamps.append(now)
                    self.token_timestamps.append((now, estimated_tokens))
                    self.daily_count += 1
                    return
                
                # Longer sleep with more jitter for Railway
                sleep_for = wait + random.uniform(1, 3)  # 1-3 seconds additional jitter
            
            print(f"RateLimiter sleeping for {sleep_for:.2f}s to respect Gemini RPM/TPM limits")
            time.sleep(sleep_for)


# Module-level rate limiter
_RATE_LIMITER = RateLimiter()

# Per-process concurrency cap
_MAX_CONCURRENCY = int(os.getenv("GEMINI_EMBEDDING_CONCURRENCY", 1))  # Reduced from 2
_embed_semaphore = threading.Semaphore(_MAX_CONCURRENCY)


def _get_retry_after(exc: Exception) -> float | None:
    """Attempt to extract a Retry-After (seconds) from common exception shapes."""
    for attr in ("response", "http_response", "resp", "raw_response"):
        resp = getattr(exc, attr, None)
        if resp is None:
            continue
        headers = getattr(resp, "headers", None) or getattr(resp, "header", None)
        if headers:
            for k in headers:
                if str(k).lower() == "retry-after":
                    try:
                        return float(headers[k])
                    except Exception:
                        try:
                            dt = datetime.fromisoformat(headers[k])
                            return max(0.0, (dt - datetime.utcnow()).total_seconds())
                        except Exception:
                            return None
    
    # Fallback: inspect message
    s = str(exc).lower()
    import re
    m = re.search(r"retry-after\D*(\d+)", s)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def _safe_call_with_semaphore(fn, *args, **kwargs):
    """Acquire the per-process semaphore, call the function with retries, release semaphore."""
    acquired = _embed_semaphore.acquire(timeout=300)
    if not acquired:
        raise RuntimeError("Could not acquire embed semaphore - too many concurrent embed requests")
    try:
        return _call_with_retries(fn, *args, **kwargs)
    finally:
        try:
            _embed_semaphore.release()
        except Exception:
            pass


# Updated retry function with better error handling

def _call_with_retries(fn, *args, max_attempts=5, base_backoff=3.0, **kwargs):  # More attempts, longer backoff
    """Call `fn(*args, **kwargs)` with retries on transient errors."""
    attempt = 0
    while True:
        attempt += 1
        try:
            result = fn(*args, **kwargs)
            _RATE_LIMITER.mark_success()  # Mark successful request
            return result
        except Exception as e:
            msg = str(e).lower()
            
            # Check for quota exhaustion specifically
            if any(tok in msg for tok in ("resource has been exhausted", "quota exceeded", "quota exhausted", "insufficient quota")):
                print(f"Quota exhausted detected: {e}")
                _RATE_LIMITER.mark_quota_exhausted(24)
                raise RuntimeError("API quota exhausted. Please check your Google AI Studio quota limits and billing.") from e
            
            # Check for authentication/permission errors (non-retryable)
            if any(tok in msg for tok in ("unauthorized", "forbidden", "invalid api key", "permission denied")):
                print(f"Authentication/permission error (non-retryable): {e}")
                raise RuntimeError("API authentication error. Please check your GEMINI_API_KEY.") from e
            
            # Check for other retryable errors
            retryable = False
            if any(tok in msg for tok in ("429", "rate limit", "rate_limit", "too many requests", "temporarily unavailable", "unavailable", "deadlineexceeded", "deadline exceeded")):
                retryable = True
                _RATE_LIMITER.mark_failure()  # Track failure
            if any(tok in msg for tok in ("timeout", "timed out", "connection reset", "connection aborted", "service unavailable", "internal error")):
                retryable = True
                
            if not retryable or attempt >= max_attempts:
                _RATE_LIMITER.mark_failure()  # Track failure
                if attempt >= max_attempts:
                    print(f"Max attempts ({max_attempts}) reached. Last error: {e}")
                raise
            
            # Handle server-provided Retry-After
            retry_after = _get_retry_after(e)
            if retry_after and retry_after > 0:
                sleep_time = min(retry_after + 2.0, 300)  # Add 2s buffer instead of 1s
                print(f"Rate limited (attempt {attempt}/{max_attempts}): {e}; waiting {sleep_time:.1f}s")
                time.sleep(sleep_time)
                continue
            
            # More conservative exponential backoff
            backoff = base_backoff * (3 ** (attempt - 1))  # Use 3x instead of 2x
            jitter = random.uniform(1, 2)  # More jitter
            sleep_time = min(backoff + jitter, 300)
            print(f"Transient error (attempt {attempt}/{max_attempts}): {e}; retrying in {sleep_time:.2f}s")
            time.sleep(sleep_time)

def _extract_from_dict_embedding_field(d: Dict[str, Any]):
    """Handle dict shapes like {'embedding': [...]} or {'embedding': [[...], [...]]}"""
    emb = d.get('embedding')
    if emb is None:
        return None
    if isinstance(emb, list):
        if len(emb) and isinstance(emb[0], (int, float)):
            return [emb]
        if len(emb) and isinstance(emb[0], (list, tuple)):
            return [list(inner) for inner in emb]
        return [list(emb)]
    if hasattr(emb, 'embedding'):
        val = getattr(emb, 'embedding')
        if isinstance(val, list) and len(val) and isinstance(val[0], (list, tuple)):
            return [list(inner) for inner in val]
        return [list(val)]
    return None


def _try_embed_via_models(contents: List[str]):
    """Try embedding via modern client.models approach"""
    try:
        estimated = sum(_estimate_tokens_for_text(t) for t in contents)
        _RATE_LIMITER.acquire(estimated)
        res = _safe_call_with_semaphore(
            client.models.embed_content,
            model=_GEMINI_MODEL, 
            contents=contents, 
            config={"task_type": "SEMANTIC_SIMILARITY"}
        )

        # Parse response
        if hasattr(res, 'embeddings'):
            return [getattr(e, 'values', getattr(e, 'embedding', e)) for e in res.embeddings]
        elif isinstance(res, dict) and 'embeddings' in res:
            return [(it.get('values') or it.get('embedding') or it) for it in res['embeddings']]
        
    except Exception as e:
        print("embed via client.models failed:", e)
        if "resource has been exhausted" in str(e).lower():
            raise  # Re-raise quota errors immediately
    return None


def _try_embed_via_embed_content(contents: List[str]):
    """Try embedding via client.models.embed_content fallback"""
    try:
        estimated = sum(_estimate_tokens_for_text(t) for t in contents)
        _RATE_LIMITER.acquire(estimated)
        res = _safe_call_with_semaphore(
            client.models.embed_content,
            model=_GEMINI_MODEL,
            contents=contents,
            config={"task_type": "SEMANTIC_SIMILARITY"}
        )

        if hasattr(res, 'embeddings'):
            return [getattr(e, 'values', getattr(e, 'embedding', e)) for e in res.embeddings]
        elif isinstance(res, dict) and 'embeddings' in res:
            return [(it.get('values') or it.get('embedding') or it) for it in res['embeddings']]
            
    except Exception as e:
        print("embed via client.models.embed_content failed:", e)
        if "resource has been exhausted" in str(e).lower():
            raise
    return None


def _try_embed_via_embed_content_direct(contents: List[str]):
    """Try embedding via direct client calls for each content"""
    try:
        results = []
        for content in contents:
            estimated = _estimate_tokens_for_text(content)
            _RATE_LIMITER.acquire(estimated)
            res = _safe_call_with_semaphore(
                client.models.embed_content,
                model=_GEMINI_MODEL,
                contents=[content],
                config={"task_type": "SEMANTIC_SIMILARITY"}
            )
            if hasattr(res, 'embeddings') and len(res.embeddings) > 0:
                embedding = res.embeddings[0]
                if hasattr(embedding, 'values'):
                    results.append(embedding.values)
                else:
                    results.append(embedding)
            else:
                print(f"No embedding found in result: {type(res)}")
                return None
        return results
    except Exception as e:
        print("embed via direct client calls failed:", e)
        if "resource has been exhausted" in str(e).lower():
            raise
    return None


def embed_text(text: str) -> list:
    """Embed a single text string"""
    embs = embed_texts([text])
    return embs[0]



# Replace your _embed_single_batch function with this debug version:

# Modern embedding function for latest Google Generative AI API
def _embed_single_batch_modern(texts: List[str]) -> List[list]:
    """Embedding function using the latest Google Generative AI API."""
    if not texts:
        return []
    
    print(f"[Embedding] Processing {len(texts)} texts with modern API...")
    
    try:
        # Use the modern google.genai.Client API
        estimated = sum(_estimate_tokens_for_text(t) for t in texts)
        _RATE_LIMITER.acquire(estimated)
        
        # Call the modern API
        result = client.models.embed_content(
            model=_GEMINI_MODEL,
            contents=texts,
            config={
                "task_type": "SEMANTIC_SIMILARITY",
                "output_dimensionality": 768  # Optional: reduce dimensionality for efficiency
            }
        )
        
        print(f"[Embedding] Modern API response type: {type(result)}")
        
        if hasattr(result, 'embeddings'):
            embeddings = []
            for i, embedding in enumerate(result.embeddings):
                if hasattr(embedding, 'values'):
                    embeddings.append(list(embedding.values))
                    print(f"[Embedding] Processed embedding {i+1}: {len(embedding.values)} dimensions")
                else:
                    raise RuntimeError(f"Embedding {i} has no 'values' attribute")
            
            print(f"[Embedding] Successfully processed {len(embeddings)} embeddings with modern API")
            return embeddings
        else:
            raise RuntimeError(f"No 'embeddings' attribute in result: {type(result)}")
            
    except Exception as e:
        print(f"[Embedding] Error in modern embedding: {e}")
        import traceback
        traceback.print_exc()
        raise

# Replace _embed_single_batch with the modern version
def _embed_single_batch(texts: List[str]) -> List[list]:
    return _embed_single_batch_modern(texts)


# Also add this debug function to test a simple API call:
def debug_simple_embedding_test():
    """Test the simplest possible embedding call"""
    print("🔍 Testing simple embedding call...")
    try:
        print(f"🔍 API Key present: {'GEMINI_API_KEY' in os.environ}")
        print(f"🔍 API Key starts with: {os.getenv('GEMINI_API_KEY', '')[:10]}...")
        
        # Test with modern API
        print("🔍 Testing with modern API (google.genai.Client)")
        result = client.models.embed_content(
            model=_GEMINI_MODEL,
            contents=["Hello world"],
            config={"task_type": "SEMANTIC_SIMILARITY"}
        )
        print(f"🔍 Modern API result type: {type(result)}")
        if hasattr(result, 'embeddings'):
            print(f"🔍 Embedding values length: {len(result.embeddings[0].values) if result.embeddings else 0}")
        return True
    except Exception as e:
        print(f"🔍 Simple test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Call this at the start of your processing:
# debug_simple_embedding_test()
# 2. In your embed_texts function, add delays between batches:

def embed_texts(texts: List[str], max_batch_tokens: int = 1000) -> List[list]:
    """Embed multiple text strings, splitting into batches by token count."""
    if not texts:
        return []
    
    batches = []
    current_batch = []
    current_tokens = 0
    
    for text in texts:
        text_tokens = _estimate_tokens_for_text(text)
        if current_tokens + text_tokens > max_batch_tokens and current_batch:
            batches.append(current_batch)
            current_batch = [text]
            current_tokens = text_tokens
        else:
            current_batch.append(text)
            current_tokens += text_tokens
    
    if current_batch:
        batches.append(current_batch)
    
    print(f"Processing {len(batches)} batches of embeddings")
    
    all_embeddings = []
    for i, batch in enumerate(batches):
        print(f"Processing batch {i+1}/{len(batches)}")
        
        batch_embeddings = _embed_single_batch(batch)
        all_embeddings.extend(batch_embeddings)
        
        # Add delay between batches (except after the last one)
        if i < len(batches) - 1:
            sleep_time = 1.5  # 1.5 seconds between batches
            print(f"Waiting {sleep_time}s before next batch...")
            time.sleep(sleep_time)
    
    return all_embeddings


# --- Summarization (real) ---
import re, json

_SUMMARY_MODEL = os.getenv("GEMINI_SUMMARY_MODEL", "gemini-2.5-flash")
_MAX_SUMMARY_CHUNKS = int(os.getenv("SUMMARY_MAX_CHUNKS", 12))   # cap cost/time
_PER_CHUNK_WORDS = int(os.getenv("SUMMARY_PER_CHUNK_WORDS", 28)) # brevity target

def _summarize_one_chunk(text: str) -> str:
    """Return one concise bullet (string) for a chunk."""
    if not text.strip():
        return ""
    
    prompt = (
        "Summarize the following legal text as ONE short bullet "
        f"(≤{_PER_CHUNK_WORDS} words). Focus on obligations, fees, "
        "dates/renewal/termination, liabilities, and privacy. "
        "No preamble, no numbering, no quotes—return only the bullet text.\n\n"
        f"---\n{text}\n---"
    )
    try:
        config = GenerateContentConfig(temperature=0.3)  # Creative for summaries
        resp = client.models.generate_content(
            model=_SUMMARY_MODEL,
            contents=prompt,
            config=config
        )
        bullet = (resp.text or "").strip()
        bullet = re.sub(r"^[\-•\s]+", "", bullet)  # strip leading bullet chars
        return bullet
    except Exception as e:
        # Fall back to a trimmed snippet so we never return the old placeholder
        return text.strip()[:120] + ("…" if len(text) > 120 else "")

def _infer_risks_from_bullets(bullets: list[str]) -> list[dict]:
    """Ask the model for up to 3 potential risks based on bullets; best-effort JSON."""
    bullets_clean = "\n".join(f"- {b}" for b in bullets if b)
    if not bullets_clean.strip():
        return []
    
    prompt = (
        "Given these bullets extracted from a legal document, identify up to 3 potential risks. "
        "Return ONLY a JSON array, where each item has keys 'label' and 'explanation'.\n\n"
        f"{bullets_clean}\n\nJSON:"
    )
    try:
        config = GenerateContentConfig(temperature=0.2)  # Consistent for analysis
        resp = client.models.generate_content(
            model=_SUMMARY_MODEL,
            contents=prompt,
            config=config
        )
        txt = (resp.text or "").strip()
        # try to extract JSON array
        m = re.search(r"\[\s*{.*}\s*\]", txt, flags=re.S)
        risks = json.loads(m.group(0)) if m else json.loads(txt)
        # basic shape cleanup
        out = []
        for r in risks:
            out.append({
                "label": r.get("label", "").strip()[:80],
                "explanation": r.get("explanation", "").strip()[:400],
            })
        return out[:3]
    except Exception:
        return []

def generate_summary(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarize each chunk with Gemini (1 bullet per chunk) and optionally infer risks.
    Returns: {"bullets": [str], "risks": [{"label": str, "explanation": str}, ...]}
    """
    if not chunks:
        return {"bullets": [], "risks": []}

    # Limit how many chunks we summarize to control cost/latency
    subset = chunks[:_MAX_SUMMARY_CHUNKS]

    bullets: list[str] = []
    for c in subset:
        b = _summarize_one_chunk(c.get("text", ""))
        if b:
            bullets.append(b)

    risks = _infer_risks_from_bullets(bullets)
    return {"bullets": bullets, "risks": risks}
