import requests
import pandas as pd
from io import BytesIO
from PIL import Image
import numpy as np
import re
from typing import Optional, List
import os
import json
import google.generativeai as genai
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import warnings
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
from datetime import datetime
import logging
import gc

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration (env-driven for production)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyA2CkFU6O3eI7CGj1B60naPfZybuncpKx4")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    logger.warning("GOOGLE_API_KEY not set. Gemini functions will not work.")

# Rate limiting
_last_gemini_call_time = 0
_gemini_rate_limit_delay = 2.0

def load_image_optimized(image_path_or_url: str) -> Image.Image:
    """Load an image from URL or local path and return PIL Image with memory optimization."""
    if image_path_or_url.startswith("http"):
        resp = requests.get(image_path_or_url, timeout=60, stream=True)
        resp.raise_for_status()
        image = Image.open(BytesIO(resp.content))
        
        # Reduce image size to save memory
        if image.size[0] > 1000 or image.size[1] > 1400:
            image = image.resize((800, 1200), Image.Resampling.LANCZOS)
        
        return image.convert("RGB")
    return Image.open(image_path_or_url).convert("RGB")

def extract_articles_with_gemini(image: Image.Image) -> list:
    """Send the page image to Gemini and ask for Gujarati headline/content pairs with rate limiting."""
    global _last_gemini_call_time
    
    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY not set. Gemini extraction disabled.")
        return []

    # Rate limiting
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
        "- content: short Gujarati summary (40-50 words)\n"
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

        # Extract JSON
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

def _parse_date_from_navgujarat_url(issue_url: str) -> str:
    """Parse date from a NavGujarat Samay issue URL.
    Supports both formats: 15-SEPT-2025 and 21092025 (DDMMYYYY)
    Returns ISO date 'YYYY-MM-DD' if possible, else ''.
    """
    try:
        # Extract last path segment (e.g., '15-SEPT-2025' or '21092025')
        path = urlparse(issue_url).path
        parts = [p for p in path.split('/') if p]
        date_token = parts[-1] if parts else ''
        if not date_token:
            # Fallback to fragment
            date_token = urlparse(issue_url).fragment

        # Case 1: Named month formats like 15-SEPT-2025
        month_map = {
            'JAN': 1, 'JANUARY': 1,
            'FEB': 2, 'FEBRUARY': 2,
            'MAR': 3, 'MARCH': 3,
            'APR': 4, 'APRIL': 4,
            'MAY': 5,
            'JUN': 6, 'JUNE': 6,
            'JUL': 7, 'JULY': 7,
            'AUG': 8, 'AUGUST': 8,
            'SEP': 9, 'SEPT': 9, 'SEPTEMBER': 9,
            'OCT': 10, 'OCTOBER': 10,
            'NOV': 11, 'NOVEMBER': 11,
            'DEC': 12, 'DECEMBER': 12,
        }
        m = re.search(r"(\d{1,2})-([A-Za-z]+)-(\d{4})", date_token)
        if m:
            day = int(m.group(1))
            mon_name = m.group(2).upper()
            year = int(m.group(3))
            month = month_map.get(mon_name)
            if not month:
                return ""
            dt = datetime(year, month, day)
            return dt.strftime("%Y-%m-%d")

        # Case 2: Numeric DDMMYYYY like 21092025
        m2 = re.search(r"^(\d{2})(\d{2})(\d{4})$", date_token)
        if m2:
            day = int(m2.group(1))
            month = int(m2.group(2))
            year = int(m2.group(3))
            dt = datetime(year, month, day)
            return dt.strftime("%Y-%m-%d")

        return ""
    except Exception:
        return ""

def get_ahmedabad_edition_url(base_url: str = "https://epaper.navgujaratsamay.com/") -> Optional[str]:
    """Extract the Ahmedabad edition URL from the main page dynamically."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1280,1696")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    
    # Explicit binary locations (Docker/Render friendly)
    chrome_bin = os.getenv("CHROME_BIN", "/usr/bin/chromium")
    if os.path.exists(chrome_bin):
        chrome_options.binary_location = chrome_bin

    driver = None
    try:
        logger.info("Getting Ahmedabad edition URL from main page...")
        
        # Prefer system chromedriver; fallback to webdriver-manager
        chromedriver_path = os.getenv("CHROMEDRIVER")
        if chromedriver_path and os.path.exists(chromedriver_path):
            service = Service(executable_path=chromedriver_path)
        else:
            logger.info("Using webdriver-manager to obtain driver...")
            driver_path = ChromeDriverManager().install()
            service = Service(executable_path=driver_path)

        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.get(base_url)
        
        # Wait for the page to load and find the card-box with href
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".card-box a"))
        )
        
        # Find the first (and only) edition link
        edition_link = driver.find_element(By.CSS_SELECTOR, ".card-box a")
        href = edition_link.get_attribute("href")
        
        if href:
            logger.info(f"Found edition URL: {href}")
            return href
        else:
            logger.error("No href found in edition card")
            return None
            
    except Exception as e:
        logger.error(f"Error getting edition URL: {e}")
        return None
    finally:
        if driver:
            driver.quit()

def fetch_navgujarat_page_image_urls(issue_url: str, max_pages: int = 18, wait_sec: int = 10) -> List[str]:
    """Given a NavGujarat Samay issue URL, iterate pages and collect page image URLs."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1280,1696")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    
    # Explicit binary locations (Docker/Render friendly)
    chrome_bin = os.getenv("CHROME_BIN", "/usr/bin/chromium")
    if os.path.exists(chrome_bin):
        chrome_options.binary_location = chrome_bin

    driver = None
    try:
        logger.info("Initializing Chrome driver for NavGujarat Samay...")
        
        # Prefer system chromedriver; fallback to webdriver-manager
        chromedriver_path = os.getenv("CHROMEDRIVER")
        if chromedriver_path and os.path.exists(chromedriver_path):
            service = Service(executable_path=chromedriver_path)
        else:
            logger.info("Using webdriver-manager to obtain driver...")
            driver_path = ChromeDriverManager().install()
            service = Service(executable_path=driver_path)

        driver = webdriver.Chrome(service=service, options=chrome_options)

        # Base without fragment
        parsed = urlparse(issue_url)
        base = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

        urls: List[str] = []
        seen: set[str] = set()
        consecutive_fail = 0

        for page in range(1, max_pages + 1):
            nav_url = f"{base}#page/{page}/1"
            logger.info(f"Loading page {page}: {nav_url}")
            try:
                driver.get(nav_url)
                # Wait for the main thumbnail image to appear
                WebDriverWait(driver, wait_sec).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "#left-bgThumbdiv img"))
                )
                time.sleep(1)  # Small buffer for src to load
                img_el = driver.find_element(By.CSS_SELECTOR, "#left-bgThumbdiv img")
                src = img_el.get_attribute("src")
                if not src:
                    logger.info(f"No image src for page {page}")
                    consecutive_fail += 1
                    if consecutive_fail >= 2:
                        break
                    continue
                src = src.strip()
                logger.info(f"Found image: {src}")
                if src in seen:
                    logger.info("Image already seen - assuming end of pages.")
                    break
                seen.add(src)
                urls.append(src)
                consecutive_fail = 0
            except Exception as e:
                logger.error(f"Error loading/extracting page {page}: {e}")
                consecutive_fail += 1
                if consecutive_fail >= 2:
                    break
                continue

        logger.info(f"Total valid image URLs found: {len(urls)}")
        return urls
    except Exception as e:
        logger.error(f"Error with Selenium: {e}")
        return []
    finally:
        if driver:
            driver.quit()

def process_navgujarat_single_page(img_url: str, date_str: str, page_num: int) -> List[dict]:
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
                "edition": "ahmedabad",  # Only Ahmedabad edition supported
                "city": article.get("city", ""),
                "district": article.get("district", ""),
                "media_link": img_url,
                "news_paper_name": "NavGujarat Samay",
                "headline": article.get("headline", ""),
                "content": article.get("content", ""),
                "sentiment": article.get("sentiment", "")
            })
        
        return formatted_articles
        
    except Exception as e:
        logger.error(f"Error processing page {page_num}: {e}")
        return []

def process_navgujarat_ahmedabad(
    issue_url: Optional[str] = None,
    date_yyyy_mm_dd: Optional[str] = None,
    max_pages: int = 18,
) -> Optional[pd.DataFrame]:
    """Process NavGujarat Samay (Ahmedabad) issue and extract articles per page.
    
    If issue_url is None, will automatically get today's edition from main page.
    """
    # If no issue URL provided, get it from main page
    if not issue_url:
        logger.info("No issue URL provided, getting from main page...")
        issue_url = get_ahmedabad_edition_url()
        if not issue_url:
            logger.error("Failed to get edition URL from main page")
            return None
    
    if not date_yyyy_mm_dd:
        date_yyyy_mm_dd = _parse_date_from_navgujarat_url(issue_url) or ""
    if not date_yyyy_mm_dd:
        logger.warning("Could not parse date from URL; 'date' column will be empty.")

    logger.info("Starting NavGujarat Samay (Ahmedabad) extraction...")
    logger.info(f"Issue URL: {issue_url}")

    page_img_urls = fetch_navgujarat_page_image_urls(issue_url, max_pages=max_pages)
    if not page_img_urls:
        logger.warning("No page images found. Check URL/date or site structure.")
        return None

    all_articles = []
    for page_num, img_url in enumerate(page_img_urls, start=1):
        articles = process_navgujarat_single_page(img_url, date_yyyy_mm_dd, page_num)
        if articles:
            all_articles.extend(articles)
        # Small delay between pages
        time.sleep(1)

    if not all_articles:
        logger.warning("No data collected for the given issue.")
        return None

    # Create DataFrame with optimized memory usage
    columns = [
        "date", "page_number", "edition", "city", "district", 
        "media_link", "news_paper_name", "headline", "content", "sentiment"
    ]
    
    df = pd.DataFrame(all_articles, columns=columns)
    logger.info(f"Total rows: {len(df)} | Pages processed: {len(page_img_urls)}")
    return df

# Public API function
def process_navgujarat(issue_url: Optional[str] = None, date_yyyy_mm_dd: Optional[str] = None, max_pages: int = 18) -> Optional[pd.DataFrame]:
    """Main function to process NavGujarat Samay newspaper (Ahmedabad edition only)."""
    return process_navgujarat_ahmedabad(issue_url, date_yyyy_mm_dd, max_pages)