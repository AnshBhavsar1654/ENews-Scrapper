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
GOOGLE_API_KEY = "AIzaSyBWMBU7A-CdLoDoOdOD5tCLFayAW7sbvm8"
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
        # Resolve Chrome and Chromedriver paths
        # On Render (Docker), set CHROME_BIN=/usr/bin/chromium; locally on Windows, leave unset
        chrome_bin = os.getenv("CHROME_BIN")
        chromedriver_env = os.getenv("CHROMEDRIVER_PATH")
        # Determine chromedriver path:
        # 1) Env var CHROMEDRIVER_PATH
        # 2) In PATH
        # 3) webdriver-manager (if installed)
        chromedriver_path = chromedriver_env or which("chromedriver") or ""

        options = Options()
        # Only set binary_location if explicitly provided (e.g., in Docker/Render)
        if chrome_bin and os.path.exists(chrome_bin):
            options.binary_location = chrome_bin
        # Headless/CI friendly flags
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-software-rasterizer")
        options.add_argument("--disable-features=VizDisplayCompositor")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-infobars")
        options.add_argument("--remote-debugging-port=9222")
        options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

        if chromedriver_path and os.path.exists(chromedriver_path):
            service = Service(executable_path=chromedriver_path)
        elif chromedriver_path:  # found via PATH which()
            service = Service(executable_path=chromedriver_path)
        elif _HAS_WDM:
            # Auto-download a compatible driver locally (great for Windows dev)
            service = Service(ChromeDriverManager().install())
        else:
            raise WebDriverException(
                "Chromedriver not found. Install chromedriver, set CHROMEDRIVER_PATH, or install webdriver-manager."
            )
        driver = webdriver.Chrome(service=service, options=options)
        # Timeouts
        driver.set_page_load_timeout(120)
        driver.set_script_timeout(60)

        # Load page
        try:
            driver.get(url)
        except TimeoutException:
            # Stop loading and proceed with whatever is rendered
            try:
                driver.execute_script("window.stop();")
            except Exception:
                pass

        # Wait for either carousel or any epaper img
        try:
            WebDriverWait(driver, 40).until(
                EC.any_of(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".carousel-inner img")),
                    EC.presence_of_element_located((By.CSS_SELECTOR, "img[src*='epaper']"))
                )
            )
        except TimeoutException:
            logger.warning("Timed out waiting for image elements")

        # Give a brief settle time
        time.sleep(2)

        # Collect images
        elems = driver.find_elements(By.CSS_SELECTOR, ".carousel-inner img, img[src*='epaper']")
        urls: List[str] = []
        seen = set()
        for img in elems:
            src = img.get_attribute("src")
            if src and any(ext in src for ext in [".jpg", ".jpeg", ".png", ".webp"]):
                if src not in seen:
                    seen.add(src)
                    urls.append(src)
        return urls
    except (WebDriverException, Exception) as e:
        logger.error(f"Selenium fetch failed: {e}")
        return []
    finally:
        try:
            if driver is not None:
                driver.quit()
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