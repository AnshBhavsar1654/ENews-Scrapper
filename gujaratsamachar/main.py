from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uuid
import os
from datetime import datetime
import time
import psutil
import logging
from scraper import process_gujarat_samachar

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Newspaper Scraper API",
    description="API for scraping Gujarati newspaper e-papers",
    version="1.0.0"
)

# In-memory task storage
task_store: Dict[str, Dict[str, Any]] = {}
MAX_TASKS = 10  # Limit stored tasks to prevent memory issues

class ScrapeRequest(BaseModel):
    date: Optional[str] = None
    editions: List[str] = ["ahmedabad"]
    max_pages: int = 5  # Reduced for free tier

class TaskStatus(BaseModel):
    task_id: str
    status: str
    message: Optional[str] = None
    progress: Optional[Dict[str, Any]] = None
    result: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None

def get_today_date() -> str:
    """Return today's date in dd-mm-yyyy format"""
    return datetime.now().strftime("%d-%m-%Y")

def check_memory_usage() -> float:
    """Check current memory usage in MB"""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0

def cleanup_old_tasks():
    """Remove old tasks to prevent memory buildup"""
    if len(task_store) > MAX_TASKS:
        # Remove oldest tasks
        oldest_tasks = sorted(task_store.items(), key=lambda x: x[1]['created_at'])[:len(task_store) - MAX_TASKS]
        for task_id, _ in oldest_tasks:
            del task_store[task_id]
        logger.info(f"Cleaned up {len(oldest_tasks)} old tasks")

@app.get("/")
async def root():
    return {
        "message": "Newspaper Scraper API",
        "status": "active",
        "memory_usage": f"{check_memory_usage():.1f} MB"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "memory_usage_mb": check_memory_usage()
    }

@app.post("/scrape/gujarat-samachar", response_model=TaskStatus)
async def scrape_gujarat_samachar_endpoint(request: ScrapeRequest, background_tasks: BackgroundTasks):
    # Check memory before starting
    memory_usage = check_memory_usage()
    if memory_usage > 400:  # 400MB threshold for 512MB RAM
        raise HTTPException(status_code=429, detail="Server memory usage too high")
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    date_str = request.date or get_today_date()
    
    # Validate editions
    if not request.editions:
        raise HTTPException(status_code=400, detail="At least one edition is required")
    
    # Limit editions for free tier
    if len(request.editions) > 3:
        request.editions = request.editions[:3]
        logger.warning(f"Limited to 3 editions due to resource constraints")
    
    # Limit pages for free tier
    if request.max_pages > 10:
        request.max_pages = 10
        logger.warning(f"Limited to 10 pages per edition due to resource constraints")
    
    # Store initial task status
    task_store[task_id] = {
        "status": "processing",
        "message": f"Scraping started for {len(request.editions)} editions",
        "progress": {
            "editions_total": len(request.editions),
            "editions_completed": 0,
            "articles_extracted": 0
        },
        "created_at": datetime.now().isoformat(),
        "params": {
            "date": date_str,
            "editions": request.editions,
            "max_pages": request.max_pages
        }
    }
    
    # Clean up old tasks
    cleanup_old_tasks()
    
    # Add background task
    background_tasks.add_task(
        process_scraping_task,
        task_id,
        request.editions,
        date_str,
        request.max_pages
    )
    
    return TaskStatus(
        task_id=task_id,
        status="started",
        message=f"Scraping started for {len(request.editions)} editions on {date_str}",
        created_at=task_store[task_id]["created_at"]
    )

@app.get("/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    if task_id not in task_store:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_data = task_store[task_id]
    return TaskStatus(
        task_id=task_id,
        status=task_data["status"],
        message=task_data.get("message"),
        progress=task_data.get("progress"),
        result=task_data.get("result"),
        error=task_data.get("error"),
        created_at=task_data["created_at"],
        completed_at=task_data.get("completed_at")
    )

@app.get("/tasks")
async def list_tasks():
    """List all active tasks"""
    return {
        "total_tasks": len(task_store),
        "tasks": [
            {
                "task_id": task_id,
                "status": data["status"],
                "created_at": data["created_at"]
            }
            for task_id, data in task_store.items()
        ]
    }

def process_scraping_task(task_id: str, editions: List[str], date_str: str, max_pages: int):
    """Background task to process scraping"""
    try:
        logger.info(f"Starting background task {task_id}")
        
        # Update progress
        task_store[task_id]["progress"]["started_at"] = datetime.now().isoformat()
        
        # Process the scraping
        df = process_gujarat_samachar(date_str, editions, max_pages)
        
        if df is not None:
            result = df.to_dict(orient='records')
            articles_count = len(result)
        else:
            result = []
            articles_count = 0
        
        # Update task status
        task_store[task_id].update({
            "status": "completed",
            "message": f"Scraping completed successfully",
            "progress": {
                "editions_total": len(editions),
                "editions_completed": len(editions),
                "articles_extracted": articles_count,
                "memory_usage_mb": check_memory_usage()
            },
            "result": result,
            "completed_at": datetime.now().isoformat()
        })
        
        logger.info(f"Task {task_id} completed with {articles_count} articles")
        
    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}")
        task_store[task_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        })

# Add startup event to check environment
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Newspaper Scraper API")
    logger.info(f"Memory usage: {check_memory_usage():.1f} MB")
    
    if not os.getenv("GOOGLE_API_KEY"):
        logger.warning("GOOGLE_API_KEY environment variable not set!")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
