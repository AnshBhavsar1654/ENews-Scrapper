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

@app.get("/debug/chrome")
async def debug_chrome():
    """Debug endpoint to check Chrome and ChromeDriver availability"""
    import subprocess
    import os
    
    debug_info = {}
    
    # Check Chrome binary
    chrome_bin = os.getenv("CHROME_BIN", "/usr/bin/google-chrome")
    debug_info["chrome_bin_path"] = chrome_bin
    debug_info["chrome_exists"] = os.path.exists(chrome_bin)
    
    if debug_info["chrome_exists"]:
        try:
            result = subprocess.run([chrome_bin, "--version"], 
                                  capture_output=True, text=True, timeout=10)
            debug_info["chrome_version"] = result.stdout.strip()
        except Exception as e:
            debug_info["chrome_version_error"] = str(e)
    
    # Check ChromeDriver
    chromedriver_path = os.getenv("CHROMEDRIVER_PATH", "/usr/local/bin/chromedriver")
    debug_info["chromedriver_path"] = chromedriver_path
    debug_info["chromedriver_exists"] = os.path.exists(chromedriver_path)
    
    if debug_info["chromedriver_exists"]:
        try:
            result = subprocess.run([chromedriver_path, "--version"], 
                                  capture_output=True, text=True, timeout=10)
            debug_info["chromedriver_version"] = result.stdout.strip()
        except Exception as e:
            debug_info["chromedriver_version_error"] = str(e)
    
    # Check environment
    debug_info["environment"] = {
        "DISPLAY": os.getenv("DISPLAY"),
        "USER": os.getenv("USER"),
        "HOME": os.getenv("HOME"),
        "PATH": os.getenv("PATH")[:200] + "..." if os.getenv("PATH") else None
    }
    
    # Test basic Chrome launch
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        
        if debug_info["chrome_exists"]:
            options.binary_location = chrome_bin
            
        service = Service(executable_path=chromedriver_path) if debug_info["chromedriver_exists"] else None
        
        driver = webdriver.Chrome(service=service, options=options)
        driver.get("https://httpbin.org/get")
        title = driver.title
        driver.quit()
        
        debug_info["test_navigation"] = f"Success - Page title: {title}"
        
    except Exception as e:
        debug_info["test_navigation"] = f"Failed: {str(e)}"
    
    return debug_info


@app.get("/debug/test-sandesh")
async def debug_test_sandesh():
    """Test Sandesh URL accessibility"""
    import requests
    from bs4 import BeautifulSoup
    
    test_url = "https://sandesh.com/epaper/ahmedabad?date=2025-01-19"
    
    try:
        response = requests.get(test_url, timeout=30, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
        })
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        return {
            "url": test_url,
            "status_code": response.status_code,
            "content_length": len(response.text),
            "title": soup.title.string if soup.title else "No title",
            "images_found": len(soup.find_all('img')),
            "carousel_elements": len(soup.select('.carousel-inner')),
            "epaper_images": len(soup.select('img[src*="epaper"]')),
            "page_snippet": response.text[:500]
        }
    except Exception as e:
        return {
            "url": test_url,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)