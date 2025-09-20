import requests
import pandas as pd
from io import BytesIO
from PIL import Image
import re
import json
import google.generativeai as genai
import os
import time
import gc
from typing import List, Optional
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urlparse, parse_qs

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyBWMBU7A-CdLoDoOdOD5tCLFayAW7sbvm8")
GEMINI_MODEL = "gemini-2.0-flash"

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    logger.warning("GOOGLE_API_KEY not set. Gemini functions will not work.")

# Rate limiting
_last_gemini_call_time = 0
_gemini_rate_limit_delay = 2.0

def load_image_optimized(image_url: str) -> Image.Image:
    """Optimized image loading with memory management"""
    try:
        resp = requests.get(image_url, timeout=60, stream=True)
        resp.raise_for_status()
        
        # Load and optimize image
        image = Image.open(BytesIO(resp.content))
        
        # Reduce image size to save memory
        if image.size[0] > 1000 or image.size[1] > 1400:
            image = image.resize((800, 1200), Image.Resampling.LANCZOS)
        
        return image.convert("RGB")
    
    except Exception as e:
        logger.error(f"Failed to load image {image_url}: {e}")
        raise

def extract_articles_with_gemini(image: Image.Image) -> list:
    """Extract articles from image using Gemini with enhanced rate limiting"""
    global _last_gemini_call_time
    
    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY not configured")
        return []
    
    # Strict rate limiting
    current_time = time.time()
    elapsed = current_time - _last_gemini_call_time
    if elapsed < _gemini_rate_limit_delay:
        sleep_time = _gemini_rate_limit_delay - elapsed
        logger.info(f"Rate limiting: sleeping for {sleep_time:.1f}s")
        time.sleep(sleep_time)
    
    _last_gemini_call_time = time.time()

    prompt = (
        "You are an expert Gujarati news analyst. Given a full-page Gujarati newspaper image, "
        "visually read the ENTIRE page and identify ALL distinct news articles (ignore advertisements, page headers/footers, and tiny blurbs). "
        "For each real article, return the following in Gujarati where applicable: \n"
        "- headline: concise Gujarati headline \n"
        "- content: short Gujarati summary(40-50 words)\n"
        "- city: the city/location mentioned for the article (Gujarati). If none is visible, return \"\".\n"
        "- district: the district for that city/location (Gujarati). If uncertain, infer reasonably; else return \"\".\n"
        "- sentiment: one of these Gujarati labels based on article tone: \"સકારાત્મક\" (positive), \"તટસ્થ\" (neutral), or \"નકારાત્મક\" (negative).\n"
        "Output only valid JSON with EXACT schema: {\"articles\":[{\"headline\":\"...\",\"content\":\"...\",\"city\":\"...\",\"district\":\"...\",\"sentiment\":\"સકારાત્મક|તટસ્થ|નકારાત્મક\"}, ...]}"
    )

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        resp = model.generate_content([prompt, image])
        out_text = getattr(resp, "text", "")
        
        if not out_text:
            logger.warning("Empty response from Gemini")
            return []

        # Extract JSON from response
        m = re.search(r"\{.*\}", out_text, re.DOTALL)
        if m:
            out_text = m.group(0)
        
        data = json.loads(out_text)
        articles = data.get("articles", [])
        
        # Normalize and validate articles
        normalized_articles = []
        for a in articles:
            headline = str(a.get("headline", "")).strip()
            content = str(a.get("content", "")).strip()
            
            # Only include articles with actual content
            if headline or content:
                normalized_articles.append({
                    "headline": headline,
                    "content": content,
                    "city": str(a.get("city", "")).strip(),
                    "district": str(a.get("district", "")).strip(),
                    "sentiment": str(a.get("sentiment", "")).strip()
                })
        
        logger.info(f"Extracted {len(normalized_articles)} articles")
        return normalized_articles
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return []
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return []

def fetch_sandesh_page_image_urls(url_with_date: str) -> List[str]:
    """Fetch a Sandesh epaper landing page for any edition and collect all page image URLs in order using Selenium."""
    # Setup Chrome options for headless mode
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    
    driver = None
    try:
        logger.info("Initializing Chrome driver...")
        driver = webdriver.Chrome(options=chrome_options)
        
        logger.info(f"Loading page: {url_with_date}")
        driver.get(url_with_date)
        
        # Wait for the carousel to load
        logger.info("Waiting for carousel to load...")
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".carousel-inner"))
        )
        
        # Additional wait for images to load
        time.sleep(3)
        
        # Find all carousel items
        logger.info("Looking for carousel items...")
        carousel_items = driver.find_elements(By.CSS_SELECTOR, ".carousel-inner .carousel-item")
        logger.info(f"Found {len(carousel_items)} carousel items")
        
        urls = []
        seen = set()
        
        for idx, item in enumerate(carousel_items):
            try:
                # Find img tag within this carousel item
                img = item.find_element(By.TAG_NAME, "img")
                src = img.get_attribute("src")
                
                if src:
                    src = src.strip()
                    logger.info(f"Carousel item {idx + 1}: {src}")
                    
                    # Validate that this is actually an image URL
                    if src and (".jpg" in src or ".jpeg" in src or ".png" in src):
                        if src not in seen:
                            seen.add(src)
                            urls.append(src)
                            logger.info(f"Added valid image URL: {src}")
                    else:
                        logger.info(f"Skipped non-image URL: {src}")
                else:
                    logger.info(f"No src attribute found in carousel item {idx + 1}")
            except Exception as e:
                logger.error(f"Error processing carousel item {idx + 1}: {e}")
        
        logger.info(f"Total valid image URLs found: {len(urls)}")
        return urls
        
    except Exception as e:
        logger.error(f"Error with Selenium: {e}")
        return []
    finally:
        if driver:
            driver.quit()

def process_sandesh_single_page(img_url: str, date_str: str, edition: str, page_num: int) -> List[dict]:
    """Process a single page and return articles"""
    try:
        logger.info(f"Processing page {page_num}, image: {img_url}")
        image = load_image_optimized(img_url)
        articles = extract_articles_with_gemini(image)
        
        # Free memory
        del image
        gc.collect()
        
        # Format results
        formatted_articles = []
        for article in articles:
            formatted_articles.append({
                "date": date_str,
                "page_number": page_num,
                "edition": edition,
                "city": article.get("city", ""),
                "district": article.get("district", ""),
                "media_link": img_url,
                "news_paper_name": "Sandesh",
                "headline": article.get("headline", ""),
                "content": article.get("content", ""),
                "sentiment": article.get("sentiment", "")
            })
        
        return formatted_articles
        
    except Exception as e:
        logger.error(f"Error processing page {page_num}: {e}")
        return []

def process_sandesh_edition(edition: str, date_str: str, max_pages: int = 5) -> List[dict]:
    """Process a single Sandesh edition with limited pages for memory safety"""
    logger.info(f"Starting Sandesh edition: {edition}")
    all_articles = []
    
    # Convert date format from DD-MM-YYYY to YYYY-MM-DD for Sandesh
    try:
        date_obj = datetime.strptime(date_str, "%d-%m-%Y")
        sandesh_date_str = date_obj.strftime("%Y-%m-%d")
    except:
        sandesh_date_str = date_str  # Fallback
    
    base_url = f"https://sandesh.com/epaper/{edition}?date={sandesh_date_str}"
    page_img_urls = fetch_sandesh_page_image_urls(base_url)
    
    if not page_img_urls:
        logger.warning(f"No page images found for edition {edition}")
        return []
    
    # Process only up to max_pages
    for page_num, img_url in enumerate(page_img_urls[:max_pages], 1):
        articles = process_sandesh_single_page(img_url, date_str, edition, page_num)
        if articles:
            all_articles.extend(articles)
        # Small delay between pages
        time.sleep(1)
    
    logger.info(f"Completed edition {edition}: {len(all_articles)} articles")
    return all_articles

def process_sandesh_multiple_editions_sequential(editions: List[str], date_str: str, max_pages: int = 5) -> pd.DataFrame:
    """Process multiple Sandesh editions sequentially to conserve memory"""
    all_articles = []
    
    for i, edition in enumerate(editions):
        try:
            logger.info(f"Processing edition {i+1}/{len(editions)}: {edition}")
            articles = process_sandesh_edition(edition, date_str, max_pages)
            all_articles.extend(articles)
            
            # Force garbage collection between editions
            gc.collect()
            time.sleep(2)  # Brief pause between editions
            
        except Exception as e:
            logger.error(f"Failed to process edition {edition}: {e}")
            continue
    
    if not all_articles:
        return None
    
    # Create DataFrame with optimized memory usage
    columns = [
        "date", "page_number", "edition", "city", "district", 
        "media_link", "news_paper_name", "headline", "content", "sentiment"
    ]
    
    return pd.DataFrame(all_articles, columns=columns)

# Newspaper-specific function
def process_sandesh(date_str: str, editions: List[str], max_pages: int = 5) -> pd.DataFrame:
    """Main function to process Sandesh newspaper"""
    return process_sandesh_multiple_editions_sequential(editions, date_str, max_pages)