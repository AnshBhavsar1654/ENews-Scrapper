import requests
import pandas as pd
from io import BytesIO
from PIL import Image
import re
import json
import google.generativeai as genai
from bs4 import BeautifulSoup
import os
import time
import gc
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash"

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    logger.warning("GOOGLE_API_KEY not set. Gemini functions will not work.")

# Rate limiting
_last_gemini_call_time = 0
_gemini_rate_limit_delay = 2.0  # Increased delay for free tier

def load_image_optimized(image_url: str) -> Image.Image:
    """Optimized image loading with memory management"""
    try:
        resp = requests.get(image_url, timeout=60, stream=True)
        resp.raise_for_status()
        
        # Load and optimize image
        image = Image.open(BytesIO(resp.content))
        
        # Reduce image size to save memory (critical for 512MB RAM)
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
    
    # Strict rate limiting for free tier
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

def get_image_url_from_page(page_url: str) -> Optional[str]:
    """Extract image URL from the e-paper page with timeout"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }
        
        resp = requests.get(page_url, headers=headers, timeout=15)
        if resp.status_code != 200:
            logger.warning(f"HTTP {resp.status_code} for {page_url}")
            return None
            
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Try multiple selectors
        selectors = ["img.epaper_page", ".rendered_img img", ".mw-100 img", "img[src*='epaper']"]
        for selector in selectors:
            img = soup.select_one(selector)
            if img and img.get("src"):
                img_url = img["src"].strip()
                if img_url.startswith("//"):
                    img_url = "https:" + img_url
                elif img_url.startswith("/"):
                    img_url = "https://epaper.gujaratsamachar.com" + img_url
                return img_url
                
        return None
        
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout fetching page: {page_url}")
        return None
    except Exception as e:
        logger.error(f"Error fetching image URL from {page_url}: {e}")
        return None

def process_single_page(page_url: str, date_str: str, edition: str, page_num: int) -> List[dict]:
    """Process a single page and return articles"""
    try:
        img_url = get_image_url_from_page(page_url)
        if not img_url:
            logger.warning(f"No image found on page {page_num}")
            return []
        
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
                "news_paper_name": "Gujarat Samachar",
                "headline": article.get("headline", ""),
                "content": article.get("content", ""),
                "sentiment": article.get("sentiment", "")
            })
        
        return formatted_articles
        
    except Exception as e:
        logger.error(f"Error processing page {page_num}: {e}")
        return []

def process_edition(edition: str, date_str: str, max_pages: int = 5) -> List[dict]:
    """Process a single edition with limited pages for memory safety"""
    logger.info(f"Starting edition: {edition}")
    all_articles = []
    base_url = f"https://epaper.gujaratsamachar.com/{edition}/{date_str}"
    
    for page_num in range(1, max_pages + 1):
        page_url = f"{base_url}/{page_num}"
        articles = process_single_page(page_url, date_str, edition, page_num)
        all_articles.extend(articles)
        
        # Stop if no articles found on page (likely end of newspaper)
        if not articles:
            logger.info(f"No articles on page {page_num}, stopping edition {edition}")
            break
            
        # Small delay between pages
        time.sleep(1)
    
    logger.info(f"Completed edition {edition}: {len(all_articles)} articles")
    return all_articles

def process_multiple_editions_sequential(editions: List[str], date_str: str, max_pages: int = 5) -> pd.DataFrame:
    """Process multiple editions sequentially to conserve memory"""
    all_articles = []
    
    for i, edition in enumerate(editions):
        try:
            logger.info(f"Processing edition {i+1}/{len(editions)}: {edition}")
            articles = process_edition(edition, date_str, max_pages)
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
def process_gujarat_samachar(date_str: str, editions: List[str], max_pages: int = 5) -> pd.DataFrame:
    """Main function to process Gujarat Samachar newspaper"""
    return process_multiple_editions_sequential(editions, date_str, max_pages)