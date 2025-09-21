from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uuid
import os
from datetime import datetime
import psutil
import logging
from scraper import process_navgujarat

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NavGujarat Samay Scraper API",
    description="API for scraping NavGujarat Samay e-paper",
    version="1.0.0"
)

# In-memory task storage
task_store: Dict[str, Dict[str, Any]] = {}
MAX_TASKS = 10

class ScrapeRequest(BaseModel):
    issue_url: str
    date: Optional[str] = None  # YYYY-MM-DD; optional as we can parse from issue URL
    max_pages: int = 10

class TaskStatus(BaseModel):
    task_id: str
    status: str
    message: Optional[str] = None
    progress: Optional[Dict[str, Any]] = None
    result: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None

# Chrome readiness flag
chrome_initialized = False

def check_memory_usage() -> float:
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except Exception:
        return 0

def cleanup_old_tasks():
    if len(task_store) > MAX_TASKS:
        oldest_tasks = sorted(task_store.items(), key=lambda x: x[1]['created_at'])[:len(task_store) - MAX_TASKS]
        for task_id, _ in oldest_tasks:
            del task_store[task_id]
        logger.info(f"Cleaned up {len(oldest_tasks)} old tasks")

@app.get("/")
async def root():
    return {
        "message": "NavGujarat Samay Scraper API",
        "status": "active",
        "memory_usage": f"{check_memory_usage():.1f} MB",
        "chrome_initialized": chrome_initialized
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "memory_usage_mb": check_memory_usage(),
        "chrome_initialized": chrome_initialized
    }

@app.get("/chrome-status")
async def chrome_status():
    return {
        "chrome_initialized": chrome_initialized,
        "status": "ready" if chrome_initialized else "not_ready",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/scrape/navgujarat", response_model=TaskStatus)
async def scrape_navgujarat_endpoint(request: ScrapeRequest, background_tasks: BackgroundTasks):
    if not chrome_initialized:
        raise HTTPException(status_code=503, detail="Chrome driver not initialized. Please try again later.")

    memory_usage = check_memory_usage()
    if memory_usage > 400:  # 400MB threshold for 512MB RAM
        raise HTTPException(status_code=429, detail="Server memory usage too high")

    task_id = str(uuid.uuid4())

    task_store[task_id] = {
        "status": "processing",
        "message": f"Scraping started",
        "progress": {
            "pages_processed": 0,
            "articles_extracted": 0
        },
        "created_at": datetime.now().isoformat(),
        "params": {
            "issue_url": request.issue_url,
            "date": request.date,
            "max_pages": request.max_pages
        }
    }

    cleanup_old_tasks()

    background_tasks.add_task(
        process_navgujarat_task,
        task_id,
        request.issue_url,
        request.date,
        request.max_pages,
    )

    return TaskStatus(
        task_id=task_id,
        status="started",
        message=f"Scraping started",
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
        completed_at=data.get("completed_at")
    )

def process_navgujarat_task(task_id: str, issue_url: str, date: Optional[str], max_pages: int):
    try:
        logger.info(f"Starting task {task_id} for NavGujarat Samay")
        task_store[task_id]["progress"]["started_at"] = datetime.now().isoformat()

        df = process_navgujarat(issue_url, date_yyyy_mm_dd=date, max_pages=max_pages)
        if df is not None:
            result = df.to_dict(orient='records')
            articles_count = len(result)
        else:
            result = []
            articles_count = 0

        task_store[task_id].update({
            "status": "completed",
            "message": "Scraping completed successfully",
            "progress": {
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

# Startup: initialize Chrome
@app.on_event("startup")
async def startup_event():
    global chrome_initialized
    logger.info("Starting NavGujarat Samay Scraper API")
    logger.info(f"Memory usage: {check_memory_usage():.1f} MB")

    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service

        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1280,1696")
        # Set binary path for containers
        chrome_bin = os.getenv("CHROME_BIN", "/usr/bin/chromium")
        if os.path.exists(chrome_bin):
            chrome_options.binary_location = chrome_bin

        chromedriver_path = os.getenv("CHROMEDRIVER", "/usr/bin/chromedriver")
        service = Service(executable_path=chromedriver_path)
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.quit()
        chrome_initialized = True
        logger.info("Chrome driver initialized successfully")

    except Exception as e:
        logger.error(f"Chrome initialization failed: {e}")
        chrome_initialized = False

    if not os.getenv("GOOGLE_API_KEY"):
        logger.warning("GOOGLE_API_KEY environment variable not set!")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
