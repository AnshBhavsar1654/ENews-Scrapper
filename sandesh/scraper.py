import requests
import pandas as pd
from io import BytesIO
from PIL import Image
import re
import json
import os
import time
import gc
from typing import List, Optional
import logging
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, WebDriverException
from shutil import which
try:
    from webdriver_manager.chrome import ChromeDriverManager  # type: ignore
    _HAS_WDM = True
except Exception:
    _HAS_WDM = False

import google.generativeai as genai

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

# Rate limiting for Gemini (friendly for free tier)
_last_gemini_call_time = 0.0
_gemini_rate_limit_delay = 2.0


def load_image_optimized(image_url: str) -> Image.Image:
    """Optimized image loading with memory management"""
    try:
        resp = requests.get(image_url, timeout=60, stream=True)
        resp.raise_for_status()
        image = Image.open(BytesIO(resp.content))
        # Resize to reduce memory usage
        if image.size[0] > 1000 or image.size[1] > 1400:
            image = image.resize((800, 1200), Image.Resampling.LANCZOS)
        return image.convert("RGB")
    except Exception as e:
        logger.error(f"Failed to load image {image_url}: {e}")
        raise


def extract_articles_with_gemini(image: Image.Image) -> list:
    """Extract articles from image using Gemini with rate limiting"""
    global _last_gemini_call_time

    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY not configured")
        return []

    # Rate limit
    now = time.time()
    elapsed = now - _last_gemini_call_time
    if elapsed < _gemini_rate_limit_delay:
        time.sleep(_gemini_rate_limit_delay - elapsed)
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
        m = re.search(r"\{.*\}", out_text, re.DOTALL)
        if m:
            out_text = m.group(0)
        data = json.loads(out_text)
        articles = data.get("articles", [])

        normalized = []
        for a in articles:
            headline = str(a.get("headline", "")).strip()
            content = str(a.get("content", "")).strip()
            if headline or content:
                normalized.append({
                    "headline": headline,
                    "content": content,
                    "city": str(a.get("city", "")).strip(),
                    "district": str(a.get("district", "")).strip(),
                    "sentiment": str(a.get("sentiment", "")).strip(),
                })
        logger.info(f"Extracted {len(normalized)} articles")
        return normalized
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return []
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return []


# -------- Sandesh-specific helpers --------

def _fetch_image_urls_static(edition_slug: str, date_yyyy_mm_dd: str) -> List[str]:
    """Try to parse image URLs from the static HTML (no JS)."""
    url = f"https://sandesh.com/epaper/{edition_slug}?date={date_yyyy_mm_dd}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        if resp.status_code != 200:
            logger.warning(f"HTTP {resp.status_code} for {url}")
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        # Common patterns
        selectors = [
            ".carousel-inner img",
            "img[src*='epaper']",
            "img[data-src*='epaper']",
        ]
        urls: List[str] = []
        seen = set()
        for sel in selectors:
            for img in soup.select(sel):
                src = img.get("src") or img.get("data-src")
                if not src:
                    continue
                src = src.strip()
                if src.startswith("//"):
                    src = "https:" + src
                elif src.startswith("/"):
                    src = "https://sandesh.com" + src
                if any(ext in src for ext in [".jpg", ".jpeg", ".png", ".webp"]):
                    if src not in seen:
                        seen.add(src)
                        urls.append(src)
        return urls
    except Exception as e:
        logger.error(f"Static parse failed: {e}")
        return []


def _fetch_image_urls_selenium(edition_slug: str, date_yyyy_mm_dd: str) -> List[str]:
    """Use Selenium to extract page image URLs from dynamic Sandesh pages."""
    url = f"https://sandesh.com/epaper/{edition_slug}?date={date_yyyy_mm_dd}"
    driver = None
    try:
        # Chrome binary and driver paths
        chrome_bin = os.getenv("CHROME_BIN", "/usr/bin/google-chrome")
        chromedriver_path = os.getenv("CHROMEDRIVER_PATH", "/usr/local/bin/chromedriver")
        
        logger.info(f"Using Chrome: {chrome_bin}")
        logger.info(f"Using ChromeDriver: {chromedriver_path}")
        
        options = Options()
        
        # Set binary location if it exists
        if os.path.exists(chrome_bin):
            options.binary_location = chrome_bin
        
        # Comprehensive Chrome arguments for cloud deployment
        chrome_args = [
            "--headless=new",
            "--no-sandbox", 
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--disable-software-rasterizer",
            "--disable-features=VizDisplayCompositor",
            "--disable-extensions",
            "--disable-plugins",
            "--disable-images",  # Speed up loading by not loading images in DOM
            "--disable-javascript",  # If the content is in static HTML
            "--disable-default-apps",
            "--disable-sync",
            "--disable-translate",
            "--hide-scrollbars",
            "--metrics-recording-only",
            "--mute-audio",
            "--no-first-run",
            "--safebrowsing-disable-auto-update",
            "--disable-ipc-flooding-protection",
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-renderer-backgrounding",
            "--disable-field-trial-config",
            "--disable-back-forward-cache",
            "--disable-hang-monitor",
            "--disable-prompt-on-repost",
            "--disable-client-side-phishing-detection",
            "--disable-component-extensions-with-background-pages",
            "--disable-default-apps",
            "--disable-extensions-http-throttling",
            "--disable-background-networking",
            "--window-size=1920,1080",
            "--start-maximized",
            "--remote-debugging-port=9222",
            "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "--single-process",  # Use single process to reduce memory usage
            "--memory-pressure-off",
            "--max_old_space_size=4096",
            "--disable-web-security",  # Only if needed for CORS
            "--disable-features=TranslateUI",
            "--disable-features=BlinkGenPropertyTrees",
            "--disable-features=VizHitTestSurfaceLayer"
        ]
        
        for arg in chrome_args:
            options.add_argument(arg)
            
        # Additional preferences for stability
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_experimental_option("prefs", {
            "profile.default_content_setting_values.notifications": 2,
            "profile.default_content_settings.popups": 0,
            "profile.managed_default_content_settings.images": 2,
        })

        # Create service
        if os.path.exists(chromedriver_path):
            service = Service(executable_path=chromedriver_path)
        elif _HAS_WDM:
            logger.info("Using WebDriverManager to install ChromeDriver")
            service = Service(ChromeDriverManager().install())
        else:
            raise WebDriverException(
                f"ChromeDriver not found at {chromedriver_path}. Install it or set CHROMEDRIVER_PATH."
            )

        # Create driver with extended timeouts
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(180)  # Increased timeout
        driver.set_script_timeout(90)
        driver.implicitly_wait(30)
        
        logger.info(f"Loading URL: {url}")
        
        # Load page with retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                driver.get(url)
                logger.info(f"Page loaded successfully on attempt {attempt + 1}")
                break
            except TimeoutException:
                logger.warning(f"Timeout loading page, attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    try:
                        driver.execute_script("window.stop();")
                    except Exception:
                        pass
                    time.sleep(5)
                else:
                    logger.error("Failed to load page after all retries")
                    return []

        # Wait for content with multiple selectors
        wait = WebDriverWait(driver, 60)  # Increased wait time
        
        selectors_to_try = [
            ".carousel-inner img",
            "img[src*='epaper']",
            "img[data-src*='epaper']",
            ".epaper img",
            ".page-image img",
        ]
        
        elements_found = False
        for selector in selectors_to_try:
            try:
                logger.info(f"Waiting for selector: {selector}")
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                elements_found = True
                logger.info(f"Found elements with selector: {selector}")
                break
            except TimeoutException:
                logger.warning(f"Timeout waiting for selector: {selector}")
                continue
        
        if not elements_found:
            logger.warning("No image elements found with any selector")
            # Try to get page source to debug
            page_source_snippet = driver.page_source[:1000] if driver.page_source else "No page source"
            logger.info(f"Page source snippet: {page_source_snippet}")
        
        # Give additional time for dynamic content
        time.sleep(10)  # Increased wait time
        
        # Try multiple selectors to collect images
        all_selectors = [
            ".carousel-inner img", 
            "img[src*='epaper']", 
            "img[data-src*='epaper']",
            ".epaper img",
            ".page-image img",
            "img[src*='sandesh']"
        ]
        
        urls = []
        seen = set()
        
        for selector in all_selectors:
            try:
                elems = driver.find_elements(By.CSS_SELECTOR, selector)
                logger.info(f"Found {len(elems)} elements with selector: {selector}")
                
                for img in elems:
                    src = None
                    # Try different attributes
                    for attr in ['src', 'data-src', 'data-lazy-src']:
                        src = img.get_attribute(attr)
                        if src:
                            break
                    
                    if src and any(ext in src.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                        if src not in seen:
                            seen.add(src)
                            urls.append(src)
                            logger.info(f"Added image URL: {src}")
                            
            except Exception as e:
                logger.warning(f"Error processing selector {selector}: {e}")
                continue
        
        logger.info(f"Total unique image URLs found: {len(urls)}")
        return urls
        
    except Exception as e:
        logger.error(f"Selenium error: {e}")
        logger.error(f"Chrome binary exists: {os.path.exists(chrome_bin)}")
        logger.error(f"ChromeDriver exists: {os.path.exists(chromedriver_path)}")
        return []
    finally:
        if driver:
            try:
                driver.quit()
                logger.info("Chrome driver closed successfully")
            except Exception as e:
                logger.error(f"Error closing driver: {e}")
        
        # Force cleanup
        try:
            import psutil
            for proc in psutil.process_iter(['pid', 'name']):
                if 'chrome' in proc.info['name'].lower():
                    proc.kill()
        except Exception:
            pass


def fetch_sandesh_page_image_urls(edition_slug: str, date_yyyy_mm_dd: str) -> List[str]:
    """Fetch Sandesh page image URLs using Selenium (required)."""
    urls = _fetch_image_urls_selenium(edition_slug, date_yyyy_mm_dd)
    if urls:
        logger.info(f"Found {len(urls)} image URLs via Selenium")
    return urls


def process_sandesh_edition(date_yyyy_mm_dd: str, edition_slug: str) -> List[dict]:
    """Process a single Sandesh edition and return list of article dicts."""
    logger.info(f"Starting Sandesh edition: {edition_slug}")
    page_img_urls = fetch_sandesh_page_image_urls(edition_slug, date_yyyy_mm_dd)
    if not page_img_urls:
        logger.warning(f"No page images found for edition {edition_slug}")
        return []

    results: List[dict] = []
    for idx, img_url in enumerate(page_img_urls, start=1):
        logger.info(f"Processing page {idx}: {img_url}")
        try:
            image = load_image_optimized(img_url)
            articles = extract_articles_with_gemini(image)
        except Exception as e:
            logger.error(f"Failed on page {idx}: {e}")
            articles = []
        finally:
            try:
                del image
            except Exception:
                pass
            gc.collect()

        if not articles:
            # Skip ad-only pages silently; continue to next
            continue

        for a in articles:
            results.append({
                "date": date_yyyy_mm_dd,
                "page_number": idx,
                "edition": edition_slug,
                "city": a.get("city", ""),
                "district": a.get("district", ""),
                "media_link": img_url,
                "news_paper_name": "Sandesh",
                "headline": a.get("headline", ""),
                "content": a.get("content", ""),
                "sentiment": a.get("sentiment", ""),
            })
        # Small delay to be polite
        time.sleep(1)

    logger.info(f"Completed edition {edition_slug}: {len(results)} articles")
    return results


def process_multiple_editions_sequential(editions: List[str], date_str: str) -> Optional[pd.DataFrame]:
    all_articles: List[dict] = []
    for i, edition in enumerate(editions):
        try:
            logger.info(f"Edition {i+1}/{len(editions)}: {edition}")
            articles = process_sandesh_edition(date_str, edition)
            all_articles.extend(articles)
            gc.collect()
            time.sleep(2)
        except Exception as e:
            logger.error(f"Failed edition {edition}: {e}")
            continue

    if not all_articles:
        return None

    columns = [
        "date", "page_number", "edition", "city", "district",
        "media_link", "news_paper_name", "headline", "content", "sentiment"
    ]
    return pd.DataFrame(all_articles, columns=columns)


# Public entry

def process_sandesh(date_str: str, editions: List[str]) -> Optional[pd.DataFrame]:
    return process_multiple_editions_sequential(editions, date_str)