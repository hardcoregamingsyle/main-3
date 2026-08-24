from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from datetime import datetime

app = FastAPI(
    title="MoE Ultra Engine API",
    description="Ultra-memory-efficient inference engine for Mixture-of-Experts models",
    version="1.0.0"
)

# Security headers middleware
class SecurityHeadersMiddleware:
    async def __call__(self, scope, receive, send):
        async def inner(message):
            if message["type"] == "http.response.start":
                message["headers"].append((b"X-Content-Type-Options", b"nosniff"))
                message["headers"].append((b"X-Frame-Options", b"DENY"))
                message["headers"].append((b"X-XSS-Protection", b"1; mode=block"))
            await send(message)
        await inner(scope, receive, send)

app.add_middleware(SecurityHeadersMiddleware)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL", "http://localhost:5173")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting configuration
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "100"))
RATE_WINDOW = int(os.getenv("RATE_WINDOW", "60"))

@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": app.version,
        "environment": os.getenv("ENVIRONMENT", "development")
    }

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "MoE Ultra Engine API",
        "version": app.version,
        "description": "Ultra-memory-efficient MoE inference engine",
        "endpoints": [
            "/health - Health check",
            "/api/v1/inference - Model inference endpoint",
            "/api/v1/models - List available models"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "3000"))
    workers = int(os.getenv("UVICORN_WORKERS", "1"))
    uvicorn.run(app, host=host, port=port, workers=workers)
