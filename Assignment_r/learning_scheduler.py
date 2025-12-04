"""
learning_scheduler.py
---------------------
Scheduler for automated learning tasks
Runs pattern learning and optimization detection on schedule
"""

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create scheduler
scheduler = BackgroundScheduler()


def learn_patterns_job():
    """Job to learn patterns from sessions"""
    try:
        logger.info("Starting scheduled pattern learning...")
        from pattern_analyzer import learn_patterns
        count = learn_patterns()
        logger.info(f"Pattern learning complete: {count} patterns created")
    except Exception as e:
        logger.error(f"Pattern learning failed: {e}")


def detect_optimizations_job():
    """Job to detect optimizations"""
    try:
        logger.info("Starting scheduled optimization detection...")
        from optimization_detector import detect_optimizations
        count = detect_optimizations()
        logger.info(f"Optimization detection complete: {count} optimizations detected")
    except Exception as e:
        logger.error(f"Optimization detection failed: {e}")


def smart_learning_job():
    """Job to run smart learning cycle"""
    try:
        logger.info("Starting scheduled smart learning...")
        from learning_orchestrator import run_smart_learning
        results = run_smart_learning()
        
        if results.get('overall_success'):
            logger.info(f"Smart learning complete: {results.get('improvements_made', 0)} improvements made")
        else:
            logger.info("Smart learning complete: No improvements made")
    except Exception as e:
        logger.error(f"Smart learning failed: {e}")


def start_scheduler():
    """Start the learning scheduler"""
    
    # Learn patterns daily at 2 AM
    scheduler.add_job(
        learn_patterns_job,
        trigger=CronTrigger(hour=2, minute=0),
        id='learn_patterns',
        name='Learn Patterns (Daily)',
        replace_existing=True
    )
    
    # Detect optimizations every 6 hours
    scheduler.add_job(
        detect_optimizations_job,
        trigger=IntervalTrigger(hours=6),
        id='detect_optimizations',
        name='Detect Optimizations (Every 6 hours)',
        replace_existing=True
    )
    
    # Smart learning every 12 hours (feedback-driven learning)
    scheduler.add_job(
        smart_learning_job,
        trigger=IntervalTrigger(hours=12),
        id='smart_learning',
        name='Smart Learning (Every 12 hours)',
        replace_existing=True
    )
    
    scheduler.start()
    logger.info("✅ Learning scheduler started")
    logger.info("   - Pattern learning: Daily at 2:00 AM")
    logger.info("   - Optimization detection: Every 6 hours")
    logger.info("   - Smart learning: Every 12 hours")
    
    return scheduler


def stop_scheduler():
    """Stop the learning scheduler"""
    if scheduler.running:
        scheduler.shutdown()
        logger.info("✅ Learning scheduler stopped")


def get_scheduler_status():
    """Get scheduler status and job information"""
    if not scheduler.running:
        return {
            "running": False,
            "jobs": []
        }
    
    jobs = []
    for job in scheduler.get_jobs():
        next_run = job.next_run_time
        jobs.append({
            "id": job.id,
            "name": job.name,
            "next_run": next_run.isoformat() if next_run else None
        })
    
    return {
        "running": True,
        "jobs": jobs
    }


def trigger_pattern_learning():
    """Manually trigger pattern learning"""
    logger.info("Manual pattern learning triggered")
    learn_patterns_job()


def trigger_optimization_detection():
    """Manually trigger optimization detection"""
    logger.info("Manual optimization detection triggered")
    detect_optimizations_job()


def trigger_smart_learning():
    """Manually trigger smart learning"""
    logger.info("Manual smart learning triggered")
    smart_learning_job()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Learning Scheduler Test")
    print("="*60)
    
    # Start scheduler
    start_scheduler()
    
    # Show status
    status = get_scheduler_status()
    print(f"\nScheduler running: {status['running']}")
    print(f"Scheduled jobs: {len(status['jobs'])}")
    
    for job in status['jobs']:
        print(f"\n  Job: {job['name']}")
        print(f"  ID: {job['id']}")
        print(f"  Next run: {job['next_run']}")
    
    print("\n" + "="*60)
    print("Scheduler is running. Press Ctrl+C to stop.")
    print("="*60)
    
    try:
        # Keep running
        import time
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        stop_scheduler()
