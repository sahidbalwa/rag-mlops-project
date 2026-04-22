"""
Module for API middleware, such as request logging.
=== FILE: api/middleware/logging.py ===
"""

import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that logs inbound requests and calculates round-trip time.
    """

    async def dispatch(self, request: Request, call_next):
        """
        Intercepts the request, passes it down the stack, and logs details on return.
        """
        start_time = time.time()
        
        logger.info(f"Incoming Request: {request.method} {request.url.path}")
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            logger.info(f"Completed Request: {request.method} {request.url.path} in {process_time:.3f}s with status {response.status_code}")
            
            # Optionally inject an X-Process-Time header
            response.headers["X-Process-Time"] = str(process_time)
            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(f"Failed Request: {request.method} {request.url.path} in {process_time:.3f}s with error: {str(e)}")
            raise
