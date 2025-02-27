"""
Token rate limiter for Groq API to manage the 6000 TPM limit
"""

import time
import threading
import logging
from datetime import datetime, timedelta
from queue import Queue, Empty

logger = logging.getLogger(__name__)

class GroqRateLimiter:
    """
    Manages token consumption rate for Groq API to stay within rate limits
    """
    
    def __init__(self, tokens_per_minute=6000, safety_margin=0.9):
        self.tpm_limit = tokens_per_minute
        # Apply safety margin to avoid edge cases
        self.effective_tpm_limit = int(tokens_per_minute * safety_margin)
        self.token_queue = Queue()
        self.usage_window = timedelta(minutes=1)
        self.usage_history = []
        self.lock = threading.Lock()
        self.paused = False
        self.pause_until = None
        
    def check_available_capacity(self, requested_tokens):
        """
        Check if there's enough capacity for the requested tokens.
        Returns (can_proceed, wait_time_seconds)
        """
        with self.lock:
            # Prune old usage records
            current_time = datetime.now()
            self.usage_history = [usage for usage in self.usage_history 
                                 if current_time - usage[0] < self.usage_window]
            
            # Calculate current usage in the window
            tokens_used = sum(usage[1] for usage in self.usage_history)
            
            # Check if we're in a pause period
            if self.paused and self.pause_until:
                if current_time < self.pause_until:
                    wait_time = (self.pause_until - current_time).total_seconds()
                    return False, wait_time
                else:
                    # Pause period is over
                    self.paused = False
                    self.pause_until = None
            
            # Check if adding these tokens would exceed our limit
            if tokens_used + requested_tokens > self.effective_tpm_limit:
                # Calculate how long to wait
                if tokens_used >= self.effective_tpm_limit:
                    # We're already at limit, wait for oldest tokens to expire
                    oldest = self.usage_history[0][0]
                    wait_time = (oldest + self.usage_window - current_time).total_seconds()
                    return False, max(0.1, wait_time)
                else:
                    # Partial wait to free up enough capacity
                    tokens_to_free = tokens_used + requested_tokens - self.effective_tpm_limit
                    # Find how many oldest records we need to remove
                    cumulative = 0
                    records_to_wait_for = 0
                    for usage in self.usage_history:
                        cumulative += usage[1]
                        records_to_wait_for += 1
                        if cumulative >= tokens_to_free:
                            break
                    if records_to_wait_for > 0:
                        wait_record = self.usage_history[records_to_wait_for - 1]
                        wait_time = (wait_record[0] + self.usage_window - current_time).total_seconds()
                        return False, max(0.1, wait_time)
            
            # We have enough capacity
            return True, 0
    
    def record_usage(self, tokens):
        """Record token usage"""
        with self.lock:
            self.usage_history.append((datetime.now(), tokens))
    
    def handle_rate_limit_error(self, retry_after_seconds=None):
        """Handle a rate limit error by pausing all requests"""
        with self.lock:
            self.paused = True
            # If we got a specific retry-after time, use that, otherwise default to 70s
            wait_time = retry_after_seconds if retry_after_seconds else 70
            self.pause_until = datetime.now() + timedelta(seconds=wait_time)
            logger.warning(f"Rate limit reached. Pausing all requests for {wait_time} seconds")
    
    def request(self, tokens, max_retries=5, base_delay=1):
        """
        Wait if necessary, then record token usage.
        Returns time waited in seconds.
        """
        for attempt in range(max_retries):
            can_proceed, wait_time = self.check_available_capacity(tokens)
            
            if can_proceed:
                self.record_usage(tokens)
                return 0
            
            # Need to wait
            if wait_time > 0:
                # Apply exponential backoff if this isn't our first attempt
                if attempt > 0:
                    # Add jitter to avoid thundering herd
                    wait_time = min(wait_time, (2 ** attempt) * base_delay)
                
                logger.info(f"Rate limit approaching. Waiting {wait_time:.2f}s before proceeding")
                time.sleep(wait_time)
        
        # If we get here, we've failed to get capacity after max retries
        raise Exception(f"Failed to get API capacity after {max_retries} attempts")


# Create a singleton instance
groq_limiter = GroqRateLimiter() 