# backend/rag_system/api/middleware.py
import time
import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start_time = time.time()
        # Log request details
        log_message = (
            f"Incoming request: {request.method} {request.url.path} "
            f"Client: {request.client.host if request.client else 'Unknown'}"
        )
        # Optionally log headers or body (be careful with sensitive data)
        # if request.method in ["POST", "PUT", "PATCH"]:
        #     try:
        #         body = await request.json()
        #         log_message += f" Body: {body}"
        #     except Exception:
        #         log_message += " Body: (could not parse as JSON)"

        logger.info(log_message)

        response = await call_next(request)

        process_time = (time.time() - start_time) * 1000  # milliseconds
        # Log response details
        logger.info(
            f"Outgoing response: {request.method} {request.url.path} "
            f"Status: {response.status_code} Duration: {process_time:.2f}ms"
        )
        response.headers["X-Process-Time"] = str(process_time)
        return response


# Example of a simple API Key Authentication Middleware (Conceptual)
# class APIKeyAuthMiddleware(BaseHTTPMiddleware):
#     def __init__(self, app, api_key_header="X-API-Key", valid_api_key="your_secret_api_key"):
#         super().__init__(app)
#         self.api_key_header = api_key_header
#         self.valid_api_key = valid_api_key

#     async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
#         if request.url.path in ["/docs", "/openapi.json", "/health"]: # Paths to exclude from auth
#             return await call_next(request)

#         api_key = request.headers.get(self.api_key_header)
#         if api_key == self.valid_api_key:
#             response = await call_next(request)
#             return response
#         else:
#             logger.warning(f"Unauthorized API access attempt for {request.url.path} from {request.client.host}")
#             return Response(content="Unauthorized: Invalid API Key", status_code=401)


# To use these middlewares, add them in `backend/rag_system/api/app.py`:
# from rag_system.api.middleware import LoggingMiddleware
# app.add_middleware(LoggingMiddleware)
# app.add_middleware(APIKeyAuthMiddleware, valid_api_key=settings.SOME_API_KEY_FROM_ENV)
