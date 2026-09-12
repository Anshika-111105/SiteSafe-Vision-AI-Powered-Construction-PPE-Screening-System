import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

from src.utils.logger import setup_logger

logger = setup_logger("api_middleware")

MAX_UPLOAD_SIZE = 10 * 1024 * 1024 # 10 MB


class RequestTrackingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 1. Generate or extract Request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        # 2. Check Payload Size
        content_length = request.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_UPLOAD_SIZE:
            return JSONResponse(
                status_code=413,
                content={
                    "error": "Payload Too Large",
                    "detail": f"Uploaded file exceeds maximum limit of {MAX_UPLOAD_SIZE // (1024*1024)}MB.",
                    "request_id": request_id,
                },
            )

        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as e:
            logger.error(f"Unhandled exception in request {request_id}: {e}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "detail": "An unexpected server error occurred during request processing.",
                    "request_id": request_id,
                },
            )

        process_time_ms = (time.perf_counter() - t0) * 1000.0

        # 3. Security & Telemetry Headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"

        logger.info(
            f"Method={request.method} Path={request.url.path} Status={response.status_code} "
            f"Latency={process_time_ms:.2f}ms ReqID={request_id}"
        )

        return response
