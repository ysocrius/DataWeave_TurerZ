"""
Minimal FastAPI Backend for AI Document Processor
Simplified version for Free tier deployment
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time
import os

# Create FastAPI app
app = FastAPI(title="AI Document Processor API - Minimal")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint for basic connectivity test"""
    return {
        "message": "AI Document Processor API - Minimal Version", 
        "status": "running",
        "version": "1.0-minimal"
    }

@app.get("/api/health")
async def health_check():
    """Simple health check for deployment"""
    return {
        "status": "ok",
        "api": "running",
        "timestamp": time.time(),
        "version": "minimal",
        "learning_enabled": False
    }

@app.get("/api/status")
async def get_status():
    """Get system status"""
    return {
        "status": "operational",
        "features": {
            "pdf_processing": "available",
            "learning_system": "disabled",
            "mongodb": "disabled"
        },
        "environment": "production-minimal"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)