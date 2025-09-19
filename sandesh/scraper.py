import requests
import pandas as pd
from io import BytesIO
from PIL import Image
import numpy as np
import re
from typing import Tuple, Optional, List
import os
import json
import google.generativeai as genai
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
import warnings
from selenium_config import create_driver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
warnings.filterwarnings('ignore')

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyBWMBU7A-CdLoDoOdOD5tCLFayAW7sbvm8")
GEMINI_MODEL = "gemini-2.0-flash"

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

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
        print("GOOGLE_API_KEY not set. Set env var or insert your key in the file.")
        return []

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

    model = genai.GenerativeModel(GEMINI_MODEL)
    try:
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
        print(f"Gemini error: {e}")
        return []


def parse_date_from_url(url: str) -> str:
    """Extract date=YYYY-MM-DD from Sandesh epaper URL query string."""
    try:
        q = parse_qs(urlparse(url).query)
        date_val = q.get("date", [""])[0]
        return date_val
    except Exception:
        return ""

# Replace the fetch_sandesh_page_image_urls function with this optimized version
def fetch_sandesh_page_image_urls(url_with_date: str) -> List[str]:
    """Fetch a Sandesh epaper landing page for any edition and collect all page image URLs in order using Selenium."""
    driver = None
    try:
        print("Initializing Chrome driver...")
        driver = create_driver()  # Use our optimized driver
        
        print(f"Loading page: {url_with_date}")
        driver.get(url_with_date)
        
        # Wait for the carousel to load with a shorter timeout
        print("Waiting for carousel to load...")
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".carousel-inner"))
        )
        
        # Additional wait for images to load
        time.sleep(3)
        
        # Find all carousel items
        print("Looking for carousel items...")
        carousel_items = driver.find_elements(By.CSS_SELECTOR, ".carousel-inner .carousel-item")
        print(f"Found {len(carousel_items)} carousel items")
        
        urls = []
        seen = set()
        
        for idx, item in enumerate(carousel_items):
            try:
                # Find img tag within this carousel item
                img = item.find_element(By.TAG_NAME, "img")
                src = img.get_attribute("src")
                
                if src and src not in seen:
                    seen.add(src)
                    urls.append(src)
                    print(f"Added image URL {idx + 1}: {src}")
                    
            except Exception as e:
                print(f"Error processing carousel item {idx + 1}: {e}")
                continue
        
        print(f"Total valid image URLs found: {len(urls)}")
        return urls
        
    except Exception as e:
        print(f"Error with Selenium: {e}")
        return []
    finally:
        if driver:
            driver.quit()
            print("Chrome driver quit successfully")

def process_sandesh_edition(date_yyyy_mm_dd: str, edition_slug: str) -> Optional[pd.DataFrame]:
    """Process a single Sandesh edition for a given date and return a DataFrame.

    - edition_slug: the URL path segment after /epaper/, e.g., "ahmedabad", "gandhinagar", "panchmahal---dahod".
    """
    base_url = f"https://sandesh.com/epaper/{edition_slug}?date={date_yyyy_mm_dd}"
    print(f"Starting Sandesh ({edition_slug}) extraction with Gemini...")
    print(f"URL: {base_url}")
    print("-" * 50)

    page_img_urls = fetch_sandesh_page_image_urls(base_url)
    if not page_img_urls:
        print("No page images found. Check URL/date or site structure.")
        return None

    rows = []
    for idx, img_url in enumerate(page_img_urls, start=1):
        print(f"\nProcessing page {idx} -> {img_url}")
        try:
            image = load_image(img_url)
        except Exception as e:
            print(f"Failed to load page image: {e}")
            continue

        articles = extract_articles_with_gemini(image)
        if not articles:
            print("No articles extracted on this page.")
        else:
            for a in articles:
                rows.append({
                    "date": date_yyyy_mm_dd,
                    "page number": idx,
                    "edition": edition_slug,
                    "city": a.get("city", "").strip(),
                    "district": a.get("district", "").strip(),
                    "media link": img_url,
                    "news paper name": "Sandesh",
                    "headline": a.get("headline", "").strip(),
                    "content": a.get("content", "").strip(),
                    "sentiment": a.get("sentiment", "").strip(),
                })
            print(f"Extracted {len(articles)} articles from page {idx}")

    if not rows:
        print("No data collected for the given edition/date.")
        return None

    columns = [
        "date",
        "page number",
        "edition",
        "city",
        "district",
        "media link",
        "news paper name",
        "headline",
        "content",
        "sentiment",
    ]
    df = pd.DataFrame(rows, columns=columns)
    print(f"Total rows: {len(df)} | Pages processed: {len(page_img_urls)}")
    return df


def process_sandesh_ahmedabad(date_yyyy_mm_dd: str, output_file: Optional[str] = None):
    """Backward-compatible helper to process only the Ahmedabad edition."""
    df = process_sandesh_edition(date_yyyy_mm_dd, "ahmedabad")
    if df is None:
        return None
    if output_file:
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"\n✓ Data saved to {output_file}")
    return df


def process_sandesh_all_editions(
    date_yyyy_mm_dd: str,
    edition_slugs: Optional[List[str]] = None,
    output_file: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Process all requested Sandesh editions for a given date and return a combined DataFrame.

    If edition_slugs is None, uses a comprehensive default list provided by the user.
    """
    if edition_slugs is None:
        edition_slugs = [
            "ahmedabad",
            "zalawad---ahmedabad-dist",
            "gandhinagar",
            "kheda",
            "mehsana",
            "sabarkantha",
            "patan",
            "banaskantha",
            "surat",
            "surat-dist",
            "valsad",
            "navsari",
            "rajkot",
            "morbi",
            "halar",
            "junagadh",
            "saurashtra",
            "bhavnagar",
            "vadodara",
            "vadodara-dist",
            "panchmahal---dahod",
            "bharuch",
            "bhuj",
        ]

    combined: List[pd.DataFrame] = []
    for slug in edition_slugs:
        print("\n" + "=" * 70)
        print(f"Processing edition: {slug}")
        df = process_sandesh_edition(date_yyyy_mm_dd, slug)
        if df is not None and not df.empty:
            combined.append(df)
        else:
            print(f"No data for edition: {slug}")

    if not combined:
        print("No data collected for any edition.")
        return None

    full_df = pd.concat(combined, ignore_index=True)
    if output_file:
        full_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"\n✓ Combined data saved to {output_file}")

    print(f"Combined total rows: {len(full_df)} across {len(combined)} editions")
    return full_df


if __name__ == "__main__":
    # Example run for ALL editions for the date
    target_date = "2025-09-15"  # YYYY-MM-DD
    output_csv = f"sandesh_epaper_{target_date}.csv"
    df_all = process_sandesh_all_editions(target_date, output_file=output_csv)
    if df_all is not None:
        print("\n" + "="*50)
        print("Sandesh (All Editions) processing completed successfully!")
        print(f"Check '{output_csv}' for the combined results")
    else:
        print("\nProcessing failed. Please check the errors above.")