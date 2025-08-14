import json
import os
import requests
from datetime import datetime, timezone
from openai import OpenAI
import csv
import os
import time


def load_config():
    config = {}
    try:
        if os.path.exists("config.json"):
            with open("config.json", "r") as f:
                config = json.load(f)
    except Exception as e:
        print(f"⚠️ Failed to load config.json: {e}")

    config["NEWSAPI_KEY"] = os.environ.get("NEWSAPI_KEY", config.get("NEWSAPI_KEY"))
    config["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", config.get("OPENAI_API_KEY"))
    return config

# ========== CONFIGURATION ==========
LOG_FILE = "btc_predictions_log.json"
NUM_HEADLINES = 20
BTC_SAME_THRESHOLD = 50  # USD
GOLD_SAME_THRESHOLD = 10  # USD

# ========== NEWS ==========
def fetch_bitcoin_headlines_newsapi(api_key, num_articles=10):
    """Fetch latest Bitcoin-related headlines from NewsAPI."""
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": "bitcoin",
        "pageSize": num_articles,
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": api_key
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"NewsAPI Error: {response.status_code}: {response.text}")

    articles = response.json().get("articles", [])
    return [article["title"] for article in articles if article.get("title")]

def fetch_gold_headlines_newsapi(api_key, num_articles=10):
    """Fetch latest Gold-related headlines from NewsAPI."""
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": "(gold OR XAU OR \"gold price\" OR bullion)",
        "pageSize": num_articles,
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": api_key
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"NewsAPI Error: {response.status_code}: {response.text}")

    articles = response.json().get("articles", [])
    return [article["title"] for article in articles if article.get("title")]

# ========== GPT ANALYSIS ==========
def gpt_headline_sentiment_classification(headlines, client, asset_name="Bitcoin"):
    """Use GPT to classify sentiment direction based on Bitcoin headlines."""
    joined = "\n".join([f"- {h}" for h in headlines])
    prompt = f"""
You are a financial analyst AI that predicts short-term {asset_name} price movement based on recent news.

Here are the latest {asset_name}-related headlines:
{joined}

Based on these, predict whether the {asset_name} price is more likely to go UP, DOWN, or stay the SAME over the next 24 hours.

Please respond in the following format exactly:

Prediction: [UP / DOWN / SAME]
Confidence: [0.0 - 1.0]
Reason: [brief explanation, 1-2 sentences]
"""
    print(prompt)

    response = client.chat.completions.create(model="gpt-4o",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3)

    text = response.choices[0].message.content
    print("\n🔍 GPT-4 Analysis:\n" + text)

    lines = text.splitlines()
    result = {"prediction": None, "confidence": None, "reason": ""}

    for line in lines:
        if line.lower().startswith("prediction"):
            result["prediction"] = line.split(":")[1].strip().upper()
        elif line.lower().startswith("confidence"):
            try:
                result["confidence"] = float(line.split(":")[1].strip())
            except:
                result["confidence"] = None
        elif line.lower().startswith("reason"):
            result["reason"] = line.split(":", 1)[1].strip()

    return result

# ========== PRICE SIMULATION ==========
def get_btc_price_data():
    """Fetch actual BTC price for yesterday and today using CoinGecko API."""
    import requests
    from datetime import datetime, timedelta

    def fetch_price_at_date(date_str):
        url = f"https://api.coingecko.com/api/v3/coins/bitcoin/history?date={date_str}&localization=false"
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"CoinGecko API error: {response.status_code} - {response.text}")
        data = response.json()
        return data["market_data"]["current_price"]["usd"]

    today = datetime.utcnow().date()
    yesterday = today - timedelta(days=1)

    yesterday_str = yesterday.strftime("%d-%m-%Y")
    today_str = today.strftime("%d-%m-%Y")

    yesterday_price = fetch_price_at_date(yesterday_str)
    today_price = fetch_price_at_date(today_str)

    return yesterday_price, today_price

import time


def http_get_with_backoff(url, params=None, headers=None, max_retries=5, base_delay=1.0):
    """GET with exponential backoff. Returns `requests.Response` or raises last error."""
    headers = headers or {}
    for attempt in range(max_retries):
        r = requests.get(url, params=params, headers=headers)
        if r.status_code == 200:
            return r
        # For 429/5xx, backoff; else raise immediately
        if r.status_code in (429, 500, 502, 503, 504):
            sleep_s = base_delay * (2 ** attempt)
            time.sleep(sleep_s)
            continue
        r.raise_for_status()
    # Exceeded retries
    raise Exception(f"HTTP error after retries: {r.status_code} - {r.text}")

def get_gold_prices_yahoo():
    """Fetch spot gold (XAUUSD) prices for the previous close and the latest close via Yahoo Finance chart API.
    Returns (yesterday_price, today_price) in USD.
    """
    url = "https://query1.finance.yahoo.com/v8/finance/chart/XAUUSD=X"
    params = {"range": "5d", "interval": "1d"}
    headers = {"User-Agent": "Mozilla/5.0 (compatible; PriceFetcher/1.0)"}
    r = http_get_with_backoff(url, params=params, headers=headers)
    data = r.json()
    result = data.get("chart", {}).get("result", [])
    if not result:
        raise Exception("Yahoo Finance response missing 'result'.")
    quote = result[0].get("indicators", {}).get("quote", [])
    if not quote:
        raise Exception("Yahoo Finance response missing 'quote' data.")
    closes = quote[0].get("close", [])
    closes = [c for c in closes if c is not None]
    if len(closes) < 2:
        raise Exception("Not enough close data returned for XAUUSD.")
    # Use the last two *valid* closes
    yesterday_price = float(closes[-2])
    today_price = float(closes[-1])
    return yesterday_price, today_price

def get_gold_prices_stooq():
    """Fallback: fetch last two daily closes for XAUUSD from Stooq (CSV)."""
    url = "https://stooq.com/q/d/l/"
    params = {"s": "xauusd", "i": "d"}
    headers = {"User-Agent": "Mozilla/5.0 (compatible; PriceFetcher/1.0)"}
    r = http_get_with_backoff(url, params=params, headers=headers)
    lines = [ln.strip() for ln in r.text.splitlines() if ln.strip()]
    # Expect header line + rows like: date,open,high,low,close,volume
    if len(lines) < 3:
        raise Exception("Stooq returned too few rows for XAUUSD.")
    # Take last two data rows
    last = lines[-1].split(",")
    prev = lines[-2].split(",")
    today_price = float(last[4])
    yesterday_price = float(prev[4])
    return yesterday_price, today_price

def get_gold_prices_alpha_vantage():
    """Optional fallback via Alpha Vantage FX_DAILY (XAU/USD). Requires ALPHAVANTAGE_KEY env."""
    key = os.environ.get("ALPHAVANTAGE_KEY")
    if not key:
        raise Exception("Missing ALPHAVANTAGE_KEY")
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "FX_DAILY",
        "from_symbol": "XAU",
        "to_symbol": "USD",
        "datatype": "json",
        "apikey": key
    }
    r = http_get_with_backoff(url, params=params)
    data = r.json()
    series = data.get("Time Series FX (Daily)")
    if not series:
        raise Exception("Alpha Vantage missing daily series")
    # Sort by date
    dates = sorted(series.keys())
    if len(dates) < 2:
        raise Exception("Alpha Vantage returned too few days")
    last = series[dates[-1]]
    prev = series[dates[-2]]
    today_price = float(last["4. close"]) 
    yesterday_price = float(prev["4. close"]) 
    return yesterday_price, today_price

def get_gold_prices():
    errors = []
    for fn in (get_gold_prices_yahoo, get_gold_prices_stooq, get_gold_prices_alpha_vantage):
        try:
            return fn()
        except Exception as e:
            errors.append(f"{fn.__name__}: {e}")
    raise Exception("All gold price providers failed: " + " | ".join(errors))

def get_actual_movement(yesterday, today, threshold=BTC_SAME_THRESHOLD):
    if abs(today - yesterday) < threshold:
        return "SAME"
    return "UP" if today > yesterday else "DOWN"

# ========== MAIN ROUTINE ==========
def run_daily_prediction_btc():
    config = load_config()
    newsapi_key = config["NEWSAPI_KEY"]
    openai_key = config["OPENAI_API_KEY"]
    client = OpenAI(api_key=openai_key)

    today = str(datetime.now(timezone.utc).date())

    try:
        headlines = fetch_bitcoin_headlines_newsapi(newsapi_key, NUM_HEADLINES)
        gpt_result = gpt_headline_sentiment_classification(headlines, client, asset_name="Bitcoin")
    except Exception as e:
        print("❌ Error fetching or analyzing news:", e)
        return

    y_price, t_price = get_btc_price_data()
    print("\nBTC price yesterday:", y_price)
    print("BTC price today:", t_price)
    actual = get_actual_movement(y_price, t_price)
    correct = gpt_result["prediction"] == actual

    print(f"\n✅ Prediction for {today}: {gpt_result['prediction']} | Actual: {actual} | Correct: {correct}")
    print("🧠 Confidence:", gpt_result["confidence"])
    print("📚 Reason:", gpt_result["reason"])

    # Log the prediction
    now = datetime.now(timezone.utc)
    log_path = "btc_predictions_log.csv"
    fieldnames = [
        "date", "time", "prediction", "confidence", "actual", "correct",
        "yesterday_price", "today_price", "reason"
    ]

    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "date": now.date().isoformat(),
            "time": now.time().strftime("%H:%M:%S"),
            "prediction": gpt_result["prediction"],
            "confidence": gpt_result["confidence"],
            "actual": actual,
            "correct": correct,
            "yesterday_price": y_price,
            "today_price": t_price,
            "reason": gpt_result["reason"]
        })

def run_daily_prediction_gold():
    config = load_config()
    newsapi_key = config["NEWSAPI_KEY"]
    openai_key = config["OPENAI_API_KEY"]
    client = OpenAI(api_key=openai_key)

    today = str(datetime.now(timezone.utc).date())

    try:
        headlines = fetch_gold_headlines_newsapi(newsapi_key, NUM_HEADLINES)
        gpt_result = gpt_headline_sentiment_classification(headlines, client, asset_name="Gold")
    except Exception as e:
        print("❌ Error fetching or analyzing gold news:", e)
        return

    try:
        y_price, t_price = get_gold_prices()
    except Exception as e:
        print("❌ Error fetching gold prices:", e)
        return

    print("\nGold price yesterday:", y_price)
    print("Gold price today:", t_price)
    actual = get_actual_movement(y_price, t_price, threshold=GOLD_SAME_THRESHOLD)
    correct = gpt_result["prediction"] == actual

    print(f"\n✅ Gold Prediction for {today}: {gpt_result['prediction']} | Actual: {actual} | Correct: {correct}")
    print("🧠 Confidence:", gpt_result["confidence"])
    print("📚 Reason:", gpt_result["reason"])

    # Log the prediction
    now = datetime.now(timezone.utc)
    log_path = "gold_predictions_log.csv"
    fieldnames = [
        "date", "time", "prediction", "confidence", "actual", "correct",
        "yesterday_price", "today_price", "reason"
    ]

    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "date": now.date().isoformat(),
            "time": now.time().strftime("%H:%M:%S"),
            "prediction": gpt_result["prediction"],
            "confidence": gpt_result["confidence"],
            "actual": actual,
            "correct": correct,
            "yesterday_price": y_price,
            "today_price": t_price,
            "reason": gpt_result["reason"]
        })

# ========== ENTRY POINT ==========
if __name__ == "__main__":
    # Bitcoin
    run_daily_prediction_btc()
    # Gold
    run_daily_prediction_gold()
