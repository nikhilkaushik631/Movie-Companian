"""
Utility classes for performance optimization
"""
import time
from collections import OrderedDict
from typing import Any, Optional, Tuple
from functools import wraps
import hashlib
import json

class LRUCache:
    """
    Least Recently Used cache with size limits and TTL support
    """
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl  # Time to live in seconds
        self.timestamps = OrderedDict()
    
    def _is_expired(self, key: str) -> bool:
        """Check if a cache entry is expired"""
        if key not in self.timestamps:
            return True
        return time.time() - self.timestamps[key] > self.ttl
    
    def _cleanup_expired(self):
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.timestamps.items()
            if current_time - timestamp > self.ttl
        ]
        for key in expired_keys:
            self.cache.pop(key, None)
            self.timestamps.pop(key, None)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key not in self.cache or self._is_expired(key):
            if key in self.cache:
                del self.cache[key]
                del self.timestamps[key]
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        self.timestamps.move_to_end(key)
        return self.cache[key]
    
    def set(self, key: str, value: Any):
        """Set value in cache"""
        current_time = time.time()
        
        if key in self.cache:
            # Update existing entry
            self.cache[key] = value
            self.timestamps[key] = current_time
            self.cache.move_to_end(key)
            self.timestamps.move_to_end(key)
        else:
            # Add new entry
            if len(self.cache) >= self.max_size:
                # Remove least recently used items until we have space
                while len(self.cache) >= self.max_size:
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                    del self.timestamps[oldest_key]
            
            self.cache[key] = value
            self.timestamps[key] = current_time
        
        # Periodic cleanup of expired entries
        if len(self.cache) % 100 == 0:
            self._cleanup_expired()
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.timestamps.clear()
    
    def size(self) -> int:
        """Get current cache size"""
        return len(self.cache)

class APIRateLimiter:
    """
    Rate limiter to prevent API exhaustion
    """
    def __init__(self):
        self.calls = {}
        self.limits = {
            "groq": (100, 60),      # 100 calls per minute
            "gemini": (60, 60),     # 60 calls per minute  
            "tmdb": (40, 10),       # 40 calls per 10 seconds
            "pinecone": (100, 60),  # 100 calls per minute
            "omdb": (1000, 86400)   # 1000 calls per day
        }
    
    async def wait_if_needed(self, api_name: str):
        """Wait if rate limit is exceeded"""
        if api_name not in self.limits:
            return
        
        max_calls, window = self.limits[api_name]
        current_time = time.time()
        
        # Initialize if not exists
        if api_name not in self.calls:
            self.calls[api_name] = []
        
        # Clean old calls outside the window
        self.calls[api_name] = [
            call_time for call_time in self.calls[api_name]
            if current_time - call_time < window
        ]
        
        # Check if we need to wait
        if len(self.calls[api_name]) >= max_calls:
            oldest_call = min(self.calls[api_name])
            wait_time = window - (current_time - oldest_call) + 0.1
            if wait_time > 0:
                import asyncio
                await asyncio.sleep(wait_time)
        
        # Record this call
        self.calls[api_name].append(current_time)

def create_cache_key(*args, **kwargs) -> str:
    """Create a consistent cache key from arguments"""
    key_data = {
        'args': args,
        'kwargs': sorted(kwargs.items())
    }
    key_string = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.md5(key_string.encode()).hexdigest()

def cached_response(cache_instance: LRUCache, ttl: int = None):
    """Decorator for caching function responses"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Create cache key
            cache_key = f"{func.__name__}:{create_cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache_instance.set(cache_key, result)
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Create cache key
            cache_key = f"{func.__name__}:{create_cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_instance.set(cache_key, result)
            return result
        
        # Return appropriate wrapper based on whether function is async
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator