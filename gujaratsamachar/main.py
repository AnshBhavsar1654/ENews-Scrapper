from flask import Flask, jsonify, request
import requests
from PIL import Image
from io import BytesIO
import pandas as pd
import numpy as np
import re
import json
import google.generativeai as genai
from bs4 import BeautifulSoup
import warnings
from datetime import datetime
import os

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Configuration
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"

# Log incoming requests immediately and completion after response
@app.before_request
def _log_request_start():
    try:
        args = dict(request.args)
    except Exception:
        args = {}
    print(f"[REQ START] {request.remote_addr} {request.method} {request.path} args={args}", flush=True)


@app.after_request
def _log_request_end(response):
    try:
        status = response.status
    except Exception:
        status = "<unknown>"
    print(f"[REQ END]   {request.remote_addr} {request.method} {request.path} -> {status}", flush=True)
    return response

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# List of available editions
AVAILABLE_EDITIONS = [
    "ahmedabad",
    "baroda",
    "surat",
    "rajkot-saurashtra",
    "bhavnagar",
    "bhuj",
    "rajkot-city",
    "kheda-anand",
    "gandhinagar",
    "mehsana",
    "sabarkantha",
    "surendranagar",
    "bharuch-panchmahal",
    "vapi-valsad",
    "bhavnagar-local",
    "patan",
    "banaskantha",
    "junagadh",
]


def load_image(image_path_or_url: str) -> Image.Image:
    """Load an image from URL or local path and return PIL Image."""
    if image_path_or_url.startswith("http"):
        resp = requests.get(image_path_or_url, timeout=60)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    return Image.open(image_path_or_url).convert("RGB")


def extract_articles_with_gemini(image: Image.Image) -> list:
    """Send the page image to Gemini and ask for Gujarati headline/content pairs."""
    if not GOOGLE_API_KEY:
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


def get_image_url_from_page(page_url: str) -> str:
    """Fetch the epaper page and extract the main image URL."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }
        resp = requests.get(page_url, headers=headers, timeout=30)
        if resp.status_code != 200:
            return None
        
        soup = BeautifulSoup(resp.text, "html.parser")
        img = soup.select_one("img.epaper_page")
        if not img:
            img = soup.select_one(".rendered_img img")
        if not img:
            img = soup.select_one(".mw-100 img")
        if img and img.get("src"):
            return img["src"].strip()
        return None
    except Exception as e:
        print(f"Error fetching image URL from {page_url}: {e}")
        return None


def process_edition(edition: str, date_str: str, max_pages: int = 18) -> dict:
    """Process a single edition for a given date."""
    base_url = f"https://epaper.gujaratsamachar.com/{edition}/{date_str}"
    rows = []
    pages_processed = 0
    errors = []

    for page_num in range(1, max_pages + 1):
        page_url = f"{base_url}/{page_num}"
        img_url = get_image_url_from_page(page_url)
        
        if not img_url:
            if page_num == 1:
                errors.append(f"No image found for page 1 of {edition}")
            break

        try:
            image = load_image(img_url)
        except Exception as e:
            errors.append(f"Failed to load image for {edition} page {page_num}: {str(e)}")
            continue

        articles = extract_articles_with_gemini(image)
        
        for a in articles:
            rows.append({
                "date": date_str,
                "page_number": page_num,
                "edition": edition,
                "city": a.get("city", "").strip(),
                "district": a.get("district", "").strip(),
                "media_link": img_url,
                "newspaper_name": "Gujarat Samachar",
                "headline": a.get("headline", "").strip(),
                "content": a.get("content", "").strip(),
                "sentiment": a.get("sentiment", "").strip(),
            })
        
        pages_processed += 1

    return {
        "edition": edition,
        "date": date_str,
        "pages_processed": pages_processed,
        "articles_count": len(rows),
        "data": rows,
        "errors": errors
    }


@app.route('/')
def home():
    """Home route with API documentation."""
    return jsonify({
        "service": "Gujarat Samachar E-paper Scraper API",
        "version": "1.0",
        "endpoints": {
            "/scrape/all": {
                "method": "GET",
                "description": "Scrape all editions for a specific date",
                "parameters": {
                    "date": "Date in dd-mm-yyyy format (required)",
                    "max_pages": "Maximum pages to scrape per edition (optional, default: 18)"
                },
                "example": "/scrape/all?date=15-09-2025"
            },
            "/scrape/edition": {
                "method": "GET",
                "description": "Scrape a specific edition for a specific date",
                "parameters": {
                    "edition": "Edition name (required)",
                    "date": "Date in dd-mm-yyyy format (required)",
                    "max_pages": "Maximum pages to scrape (optional, default: 18)"
                },
                "example": "/scrape/edition?edition=surat&date=15-09-2025"
            },
            "/editions": {
                "method": "GET",
                "description": "Get list of available editions"
            }
        },
        "available_editions": AVAILABLE_EDITIONS
    })


@app.route('/editions')
def get_editions():
    """Get list of available editions."""
    return jsonify({
        "editions": AVAILABLE_EDITIONS,
        "count": len(AVAILABLE_EDITIONS)
    })


@app.route('/scrape/edition')
def scrape_single_edition():
    """Scrape a specific edition for a particular date."""
    edition = request.args.get('edition')
    date_str = request.args.get('date')
    max_pages = int(request.args.get('max_pages', 18))

    # Validation
    if not edition:
        return jsonify({"error": "Missing required parameter: edition"}), 400
    
    if not date_str:
        return jsonify({"error": "Missing required parameter: date"}), 400
    
    if edition not in AVAILABLE_EDITIONS:
        return jsonify({
            "error": f"Invalid edition: {edition}",
            "available_editions": AVAILABLE_EDITIONS
        }), 400
    
    # Validate date format
    try:
        datetime.strptime(date_str, "%d-%m-%Y")
    except ValueError:
        return jsonify({"error": "Invalid date format. Use dd-mm-yyyy"}), 400

    try:
        result = process_edition(edition, date_str, max_pages)
        
        if not result["data"]:
            return jsonify({
                "success": False,
                "message": "No articles found",
                "result": result
            }), 404
        
        return jsonify({
            "success": True,
            "edition": edition,
            "date": date_str,
            "pages_processed": result["pages_processed"],
            "articles_count": result["articles_count"],
            "data": result["data"],
            "errors": result["errors"] if result["errors"] else None
        })
    
    except Exception as e:
        return jsonify({
            "error": f"Processing failed: {str(e)}",
            "edition": edition,
            "date": date_str
        }), 500


@app.route('/scrape/all')
def scrape_all_editions():
    """Scrape all editions for a particular date."""
    date_str = request.args.get('date')
    max_pages = int(request.args.get('max_pages', 18))
    
    # Validation
    if not date_str:
        return jsonify({"error": "Missing required parameter: date"}), 400
    
    # Validate date format
    try:
        datetime.strptime(date_str, "%d-%m-%Y")
    except ValueError:
        return jsonify({"error": "Invalid date format. Use dd-mm-yyyy"}), 400

    all_results = []
    all_data = []
    successful_editions = []
    failed_editions = []
    total_articles = 0
    
    for edition in AVAILABLE_EDITIONS:
        try:
            result = process_edition(edition, date_str, max_pages)
            
            if result["data"]:
                all_data.extend(result["data"])
                successful_editions.append(edition)
                total_articles += result["articles_count"]
                all_results.append({
                    "edition": edition,
                    "status": "success",
                    "articles_count": result["articles_count"],
                    "pages_processed": result["pages_processed"]
                })
            else:
                failed_editions.append(edition)
                all_results.append({
                    "edition": edition,
                    "status": "no_data",
                    "articles_count": 0,
                    "errors": result["errors"]
                })
        
        except Exception as e:
            failed_editions.append(edition)
            all_results.append({
                "edition": edition,
                "status": "error",
                "error": str(e)
            })
    
    return jsonify({
        "success": len(successful_editions) > 0,
        "date": date_str,
        "summary": {
            "total_editions_processed": len(AVAILABLE_EDITIONS),
            "successful_editions": len(successful_editions),
            "failed_editions": len(failed_editions),
            "total_articles": total_articles
        },
        "editions_status": all_results,
        "successful_editions": successful_editions,
        "failed_editions": failed_editions if failed_editions else None,
        "data": all_data if all_data else None
    })


@app.route('/health')
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({
        "status": "healthy",
        "service": "Gujarat Samachar Scraper",
        "gemini_configured": bool(GOOGLE_API_KEY)
    })


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)