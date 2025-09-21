import requests
import pandas as pd
from io import BytesIO
from PIL import Image
import re
from typing import Optional, List
import os
import json
import google.generativeai as genai
from urllib.parse import urlparse
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
import time
from datetime import datetime

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration (env-driven)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    logger.warning("GOOGLE_API_KEY not set. Gemini calls will be disabled.")


def load_image(image_path_or_url: str) -> Image.Image:
    """Load an image from URL or local path and return PIL Image."""
    if image_path_or_url.startswith("http"):
        resp = requests.get(image_path_or_url, timeout=60)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    return Image.open(image_path_or_url).convert("RGB")


def extract_articles_with_gemini(image: Image.Image) -> list:
    """Send the page image to Gemini and ask for Gujarati headline/content pairs.
    Returns a list of dicts: [{"headline": str, "content": str, "city": str, "district": str, "sentiment": str}, ...]
    """
    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY not set; skipping Gemini extraction.")
        return []

    prompt = (
        "You are an expert Gujarati news analyst. Given a full-page Gujarati newspaper image, "
        "visually read the ENTIRE page and identify ALL distinct news articles (ignore advertisements, page headers/footers, and tiny blurbs). "
        "For each real article, return the following in Gujarati where applicable: \n"
        "- headline: concise Gujarati headline \n"
        "- content: short Gujarati summary\n"
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
            return []

        # Extract JSON
        m = re.search(r"\{.*\}", out_text, re.DOTALL)
        if m:
            out_text = m.group(0)
        data = json.loads(out_text)
        articles = data.get("articles", [])
        # Normalize
        norm = []
        for a in articles:
            h = str(a.get("headline", "")).strip()
            c = str(a.get("content", "")).strip()
            city = str(a.get("city", "")).strip()
            district = str(a.get("district", "")).strip()
            sentiment = str(a.get("sentiment", "")).strip()
            if h or c:
                norm.append({
                    "headline": h,
                    "content": c,
                    "city": city,
                    "district": district,
                    "sentiment": sentiment
                })
        return norm
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return []


# NavGujarat Samay helpers

def _parse_date_from_navgujarat_url(issue_url: str) -> str:
    """Parse date from a NavGujarat Samay issue URL like
    https://epaper.navgujaratsamay.com/4056813/Ahmedabad/15-SEPT-2025#page/1/1
    Returns ISO date 'YYYY-MM-DD' if possible, else ''.
    """
    try:
        # Extract segment like 15-SEPT-2025
        path = urlparse(issue_url).path
        parts = [p for p in path.split('/') if p]
        # Typically: ['', '4056813', 'Ahmedabad', '15-SEPT-2025'] -> last part is date
        date_token = parts[-1] if parts else ''
        # Handle cases where there isn't a date in path, fallback to fragment
        if not re.search(r"\d{4}", date_token):
            frag = urlparse(issue_url).fragment
            date_token = frag
        # Normalize month names
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
        if not m:
            return ""
        day = int(m.group(1))
        mon_name = m.group(2).upper()
        year = int(m.group(3))
        month = month_map.get(mon_name)
        if not month:
            return ""
        dt = datetime(year, month, day)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return ""


def fetch_navgujarat_page_image_urls(issue_url: str, max_pages: int = 18, wait_sec: int = 10) -> List[str]:
    """Given a NavGujarat Samay issue URL, iterate pages and collect page image URLs."""
    # Setup Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1280,1696")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36")
    # Explicit binary locations (Docker/Render friendly)
    chrome_bin = os.getenv("CHROME_BIN", "/usr/bin/chromium")
    if os.path.exists(chrome_bin):
        chrome_options.binary_location = chrome_bin

    driver = None
    try:
        logger.info("Initializing Chrome driver for NavGujarat Samay...")
        chromedriver_path = os.getenv("CHROMEDRIVER", "/usr/bin/chromedriver")
        service = Service(executable_path=chromedriver_path)
        driver = webdriver.Chrome(service=service, options=chrome_options)

        # base without fragment
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
                time.sleep(1)  # small buffer for src to load
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


def process_navgujarat_ahmedabad(
    issue_url: str,
    date_yyyy_mm_dd: Optional[str] = None,
    output_file: Optional[str] = None,
    max_pages: int = 18,
) -> Optional[pd.DataFrame]:
    """Process NavGujarat Samay (Ahmedabad) issue and extract articles per page."""
    if not date_yyyy_mm_dd:
        date_yyyy_mm_dd = _parse_date_from_navgujarat_url(issue_url) or ""
    if not date_yyyy_mm_dd:
        logger.warning("Could not parse date from URL; 'date' column will be empty.")

    logger.info("Starting NavGujarat Samay (Ahmedabad) extraction with Gemini...")
    logger.info(f"Issue URL: {issue_url}")

    page_img_urls = fetch_navgujarat_page_image_urls(issue_url, max_pages=max_pages)
    if not page_img_urls:
        logger.warning("No page images found. Check URL/date or site structure.")
        return None

    rows = []
    for idx, img_url in enumerate(page_img_urls, start=1):
        logger.info(f"Processing page {idx} -> {img_url}")
        try:
            image = load_image(img_url)
        except Exception as e:
            logger.error(f"Failed to load page image: {e}")
            continue

        articles = extract_articles_with_gemini(image)
        if not articles:
            logger.info("No articles extracted on this page.")
        else:
            for a in articles:
                rows.append({
                    "date": date_yyyy_mm_dd,
                    "page_number": idx,
                    "edition": "ahmedabad",
                    "city": a.get("city", "").strip(),
                    "district": a.get("district", "").strip(),
                    "media_link": img_url,
                    "news_paper_name": "NavGujarat Samay",
                    "headline": a.get("headline", "").strip(),
                    "content": a.get("content", "").strip(),
                    "sentiment": a.get("sentiment", "").strip(),
                })

    if not rows:
        logger.warning("No data collected for the given issue.")
        return None

    columns = [
        "date",
        "page_number",
        "edition",
        "city",
        "district",
        "media_link",
        "news_paper_name",
        "headline",
        "content",
        "sentiment",
    ]
    df = pd.DataFrame(rows, columns=columns)

    if output_file:
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        logger.info(f"Data saved to {output_file}")

    logger.info(f"Total rows: {len(df)} | Pages processed: {len(page_img_urls)}")
    return df


def process_navgujarat(issue_url: str, date_yyyy_mm_dd: Optional[str] = None, max_pages: int = 18) -> Optional[pd.DataFrame]:
    """Public function to be used by API to process an issue URL."""
    return process_navgujarat_ahmedabad(issue_url, date_yyyy_mm_dd=date_yyyy_mm_dd, max_pages=max_pages)