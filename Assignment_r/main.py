#!/usr/bin/env python3
"""
Main entry point for the AI Document Processor API
This ensures proper module loading in production environments
"""

import sys
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Add current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    logger.info(f"Added {current_dir} to Python path")
    
    # Import the FastAPI app
    logger.info("Importing FastAPI app...")
    from backend import app
    logger.info("FastAPI app imported successfully")
    
    if __name__ == "__main__":
        import uvicorn
        port = int(os.environ.get("PORT", 8000))
        logger.info(f"Starting uvicorn server on port {port}")
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
        
except Exception as e:
    logger.error(f"Failed to start application: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)