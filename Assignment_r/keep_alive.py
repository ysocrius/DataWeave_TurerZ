"""
Keep-alive service to prevent Render free tier from spinning down
Pings the health endpoint every 10 minutes
"""
import requests
import time
import os
from apscheduler.schedulers.background import BackgroundScheduler
import logging

logger = logging.getLogger(__name__)

# Get the service URL from environment or use default
SERVICE_URL = os.getenv("RENDER_EXTERNAL_URL", "http://localhost:8000")

def ping_self():
    """Ping the health endpoint to keep the service alive"""
    try:
        response = requests.get(f"{SERVICE_URL}/api/health", timeout=10)
        if response.status_code == 200:
            logger.info("✅ Keep-alive ping successful")
        else:
            logger.warning(f"⚠️ Keep-alive ping returned status {response.status_code}")
    except Exception as e:
        logger.error(f"❌ Keep-alive ping failed: {e}")

def start_keep_alive():
    """Start the keep-alive scheduler"""
    scheduler = BackgroundScheduler()
    
    # Ping every 10 minutes (600 seconds)
    scheduler.add_job(
        ping_self,
        'interval',
        minutes=10,
        id='keep_alive_ping',
        replace_existing=True
    )
    
    scheduler.start()
    logger.info("🔄 Keep-alive service started (pinging every 10 minutes)")
    
    return scheduler

if __name__ == "__main__":
    # For testing
    logging.basicConfig(level=logging.INFO)
    scheduler = start_keep_alive()
    
    try:
        # Keep the script running
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
