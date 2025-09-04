# token_counter.py
import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from google import genai
from google.genai import types

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TokenCounter:
    def __init__(self):
        self._api_key = os.getenv("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        self.client = genai.Client(api_key=self._api_key)
        
        # Token usage tracking
        self.session_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_input_tokens": 0,
            "total_requests": 0,
            "session_start": datetime.now()
        }
    
    def count_tokens_before_request(self, prompt: str, model: str = "gemini-2.0-flash-exp") -> Dict[str, Any]:
        """Count tokens before making a request to estimate costs."""
        try:
            # Count tokens in the prompt
            token_response = self.client.models.count_tokens(
                model=model,
                contents=prompt
            )
            
            token_count = token_response.total_tokens
            
            # Log token count
            logger.info(f"📊 Token Count - Input: {token_count} tokens for model: {model}")
            
            # Estimate cost (approximate pricing for Gemini)
            estimated_cost = self._estimate_cost(token_count, model)
            
            return {
                "input_tokens": token_count,
                "estimated_cost_usd": estimated_cost,
                "model": model,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error counting tokens: {e}")
            return {
                "input_tokens": 0,
                "estimated_cost_usd": 0,
                "error": str(e),
                "model": model,
                "timestamp": datetime.now().isoformat()
            }
    
    def track_api_usage(self, input_tokens: int, output_tokens: int, cached_tokens: int = 0):
        """Track actual API usage after request completion."""
        # Ensure all values are integers, not None
        input_tokens = input_tokens or 0
        output_tokens = output_tokens or 0
        cached_tokens = cached_tokens or 0
        
        self.session_usage["input_tokens"] += input_tokens
        self.session_usage["output_tokens"] += output_tokens
        self.session_usage["cached_input_tokens"] += cached_tokens
        self.session_usage["total_requests"] += 1
        
        total_cost = self._calculate_session_cost()
        
        logger.info(f"💰 API Usage Update:")
        logger.info(f"   Input: {input_tokens} tokens")
        logger.info(f"   Output: {output_tokens} tokens") 
        logger.info(f"   Cached: {cached_tokens} tokens")
        logger.info(f"   Session Total Cost: ${total_cost:.4f}")
        logger.info(f"   Total Requests: {self.session_usage['total_requests']}")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get session usage summary."""
        session_duration = (datetime.now() - self.session_usage["session_start"]).total_seconds()
        total_cost = self._calculate_session_cost()
        
        return {
            "session_duration_seconds": session_duration,
            "total_input_tokens": self.session_usage["input_tokens"],
            "total_output_tokens": self.session_usage["output_tokens"],
            "total_cached_tokens": self.session_usage["cached_input_tokens"],
            "total_requests": self.session_usage["total_requests"],
            "estimated_total_cost_usd": total_cost,
            "cost_savings_from_cache_usd": self._calculate_cache_savings(),
            "session_start": self.session_usage["session_start"].isoformat()
        }
    
    def _estimate_cost(self, token_count: int, model: str) -> float:
        """Estimate cost based on token count and model."""
        # Approximate Gemini pricing (per 1K tokens)
        pricing = {
            "gemini-2.0-flash-exp": {"input": 0.000075, "output": 0.0003},
            "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
            "gemini-1.5-pro": {"input": 0.00125, "output": 0.005}
        }
        
        model_pricing = pricing.get(model, pricing["gemini-2.0-flash-exp"])
        return (token_count / 1000) * model_pricing["input"]
    
    def _calculate_session_cost(self) -> float:
        """Calculate total session cost."""
        # Approximate costs
        input_cost = (self.session_usage["input_tokens"] / 1000) * 0.000075
        output_cost = (self.session_usage["output_tokens"] / 1000) * 0.0003
        # Cached tokens are much cheaper (assume 75% discount)
        cached_cost = (self.session_usage["cached_input_tokens"] / 1000) * 0.000075 * 0.25
        
        return input_cost + output_cost + cached_cost
    
    def _calculate_cache_savings(self) -> float:
        """Calculate cost savings from using cached tokens."""
        if self.session_usage["cached_input_tokens"] == 0:
            return 0
        
        # Savings = full price - discounted price for cached tokens
        full_price = (self.session_usage["cached_input_tokens"] / 1000) * 0.000075
        discounted_price = (self.session_usage["cached_input_tokens"] / 1000) * 0.000075 * 0.25
        
        return full_price - discounted_price
    
    def log_session_summary(self):
        """Log a comprehensive session summary."""
        summary = self.get_session_summary()
        
        logger.info("📊 === TOKEN USAGE SESSION SUMMARY ===")
        logger.info(f"Duration: {summary['session_duration_seconds']:.1f} seconds")
        logger.info(f"Total Requests: {summary['total_requests']}")
        logger.info(f"Input Tokens: {summary['total_input_tokens']:,}")
        logger.info(f"Output Tokens: {summary['total_output_tokens']:,}")
        logger.info(f"Cached Tokens: {summary['total_cached_tokens']:,}")
        logger.info(f"Total Cost: ${summary['estimated_total_cost_usd']:.4f}")
        logger.info(f"Cache Savings: ${summary['cost_savings_from_cache_usd']:.4f}")
        logger.info("==========================================")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get detailed session statistics for API endpoints."""
        duration = datetime.now() - self.session_usage["session_start"]
        total_tokens = self.session_usage["input_tokens"] + self.session_usage["output_tokens"]
        
        return {
            "session_start_time": self.session_usage["session_start"],
            "total_input_tokens": self.session_usage["input_tokens"],
            "total_output_tokens": self.session_usage["output_tokens"],
            "total_cached_tokens": self.session_usage["cached_input_tokens"],
            "total_api_calls": self.session_usage["total_requests"],
            "average_tokens_per_call": total_tokens / max(self.session_usage["total_requests"], 1),
            "session_duration_seconds": duration.total_seconds(),
            "total_cost_usd": self._calculate_session_cost(),
            "cache_savings_usd": self._calculate_cache_savings()
        }
    
    def reset_session_stats(self):
        """Reset session statistics."""
        self.session_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_input_tokens": 0,
            "total_requests": 0,
            "session_start": datetime.now()
        }
        logger.info("🔄 Token counter session statistics reset")

# Global token counter instance (lazy initialization)
token_counter = None

def get_token_counter():
    """Get or create the global token counter instance."""
    global token_counter
    if token_counter is None:
        token_counter = TokenCounter()
    return token_counter
