from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uuid
import os
from datetime import datetime
import psutil
import logging
from scraper import process_sandesh

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sandesh E-paper Scraper API",
    description="API for scraping Sandesh e-paper editions",
    version="1.0.0"
)

# In-memory task storage
task_store: Dict[str, Dict[str, Any]] = {}
MAX_TASKS = 10

class ScrapeRequest(BaseModel):
    date: Optional[str] = None  # Expected format: YYYY-MM-DD
    editions: List[str] = ["ahmedabad"]
    max_pages: Optional[int] = None  # Not used for Sandesh; kept for parity

class TaskStatus(BaseModel):
    task_id: str
    status: str
    message: Optional[str] = None
    progress: Optional[Dict[str, Any]] = None
    result: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None


def get_today_date_iso() -> str:
    """Return today's date in YYYY-MM-DD format"""
    return datetime.now().strftime("%Y-%m-%d")


def check_memory_usage() -> float:
    """Check current memory usage in MB"""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0


def cleanup_old_tasks():
    if len(task_store) > MAX_TASKS:
        oldest = sorted(task_store.items(), key=lambda x: x[1]['created_at'])[:len(task_store) - MAX_TASKS]
        for task_id, _ in oldest:
            del task_store[task_id]
        logger.info(f"Cleaned up {len(oldest)} old tasks")


@app.get("/")
async def root():
    return {
        "message": "Sandesh E-paper Scraper API",
        "status": "active",
        "memory_usage_mb": f"{check_memory_usage():.1f}"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "memory_usage_mb": check_memory_usage()
    }


@app.post("/scrape/sandesh", response_model=TaskStatus)
async def scrape_sandesh_endpoint(request: ScrapeRequest, background_tasks: BackgroundTasks):
    # Memory guard for 512MB environments
    if check_memory_usage() > 400:
        raise HTTPException(status_code=429, detail="Server memory usage too high")

    task_id = str(uuid.uuid4())
    date_str = request.date or get_today_date_iso()

    if not request.editions:
        raise HTTPException(status_code=400, detail="At least one edition is required")

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
        }
    }

    cleanup_old_tasks()

    background_tasks.add_task(process_scraping_task, task_id, request.editions, date_str)

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
    data = task_store[task_id]
    return TaskStatus(
        task_id=task_id,
        status=data["status"],
        message=data.get("message"),
        progress=data.get("progress"),
        result=data.get("result"),
        error=data.get("error"),
        created_at=data["created_at"],
        completed_at=data.get("completed_at"),
    )


@app.get("/tasks")
async def list_tasks():
    return {
        "total_tasks": len(task_store),
        "tasks": [
            {
                "task_id": tid,
                "status": data["status"],
                "created_at": data["created_at"]
            }
            for tid, data in task_store.items()
        ]
    }


def process_scraping_task(task_id: str, editions: List[str], date_str: str):
    try:
        logger.info(f"Starting Sandesh task {task_id}")
        task_store[task_id]["progress"]["started_at"] = datetime.now().isoformat()

        df = process_sandesh(date_str, editions)
        if df is not None:
            result = df.to_dict(orient="records")
            articles_count = len(result)
        else:
            result = []
            articles_count = 0

        task_store[task_id].update({
            "status": "completed",
            "message": "Scraping completed successfully",
            "progress": {
                "editions_total": len(editions),
                "editions_completed": len(editions),
                "articles_extracted": articles_count,
                "memory_usage_mb": check_memory_usage(),
            },
            "result": result,
            "completed_at": datetime.now().isoformat(),
        })

        logger.info(f"Task {task_id} completed with {articles_count} articles")

    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}")
        task_store[task_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now().isoformat(),
        })


@app.on_event("startup")
async def startup_event():
    logger.info("Starting Sandesh Scraper API")
    logger.info(f"Memory usage: {check_memory_usage():.1f} MB")
    if not os.getenv("GOOGLE_API_KEY"):
        logger.warning("GOOGLE_API_KEY environment variable not set!")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
