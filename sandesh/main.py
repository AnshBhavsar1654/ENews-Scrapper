# main.py
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uuid
import pandas as pd
import os
from scraper import process_sandesh_all_editions

app = FastAPI(title="E-paper Scraper API")

# Store processing status (in production, use Redis or database)
processing_status = {}

class ProcessingRequest(BaseModel):
    date: str
    editions: Optional[List[str]] = None

@app.post("/process")
async def process_epaper(request: ProcessingRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    processing_status[job_id] = {"status": "processing", "result": None, "error": None}
    
    # Run processing in background
    background_tasks.add_task(process_task, job_id, request.date, request.editions)
    
    return {"job_id": job_id, "status": "started", "message": "Processing started"}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in processing_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return processing_status[job_id]

@app.get("/results/{job_id}")
async def get_results(job_id: str):
    if job_id not in processing_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if processing_status[job_id]["status"] != "completed":
        raise HTTPException(status_code=425, detail="Job still processing")
    
    return processing_status[job_id]["result"]

def process_task(job_id: str, process_date: str, editions: List[str] = None):
    try:
        # Process the request
        result = process_sandesh_all_editions(process_date, editions, None)
        
        # Convert to dictionary for JSON response
        if result is not None:
            processing_status[job_id] = {
                "status": "completed", 
                "result": result.to_dict(orient="records"),
                "error": None
            }
        else:
            processing_status[job_id] = {
                "status": "completed", 
                "result": [],
                "error": "No data found for the specified date"
            }
            
    except Exception as e:
        processing_status[job_id] = {
            "status": "error", 
            "result": None,
            "error": str(e)
        }