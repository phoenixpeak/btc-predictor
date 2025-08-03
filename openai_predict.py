import json
import os
import requests
from datetime import datetime, timedelta, timezone
from openai import OpenAI
import random

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
SAME_THRESHOLD = 50  # USD

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

# ========== GPT ANALYSIS ==========
def gpt_headline_sentiment_classification(headlines, client):
    """Use GPT to classify sentiment direction based on Bitcoin headlines."""
    joined = "\n".join([f"- {h}" for h in headlines])
    prompt = f"""
You are a financial analyst AI that predicts short-term Bitcoin price movement based on recent news.

Here are the latest Bitcoin-related headlines:
{joined}

Based on these, predict whether the Bitcoin price is more likely to go UP, DOWN, or stay the SAME over the next 24 hours.

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
def get_price_data():
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

def get_actual_movement(yesterday, today, threshold=SAME_THRESHOLD):
    if abs(today - yesterday) < threshold:
        return "SAME"
    return "UP" if today > yesterday else "DOWN"

# ========== LOGGING ==========
def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            return json.load(f)
    return []

def save_log(data):
    with open(LOG_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ========== MAIN ROUTINE ==========
def run_daily_prediction():
    config = load_config()
    newsapi_key = config["NEWSAPI_KEY"]
    openai_key = config["OPENAI_API_KEY"]
    client = OpenAI(api_key=openai_key)

    today = str(datetime.now(timezone.utc).date())

    try:
        headlines = fetch_bitcoin_headlines_newsapi(newsapi_key, NUM_HEADLINES)
        gpt_result = gpt_headline_sentiment_classification(headlines, client)
    except Exception as e:
        print("❌ Error fetching or analyzing news:", e)
        return

    y_price, t_price = get_price_data()
    print("\nBTC price yesterday:", y_price)
    print("BTC price today:", t_price)
    actual = get_actual_movement(y_price, t_price)
    correct = gpt_result["prediction"] == actual

    entry = {
        "date": today,
        "headlines": headlines,
        "prediction": gpt_result["prediction"],
        "confidence": gpt_result["confidence"],
        "reason": gpt_result["reason"],
        "yesterday_price": y_price,
        "today_price": t_price,
        "actual": actual,
        "correct": correct
    }

    logs = load_log()
    logs.append(entry)
    save_log(logs)

    print(f"\n✅ Prediction for {today}: {gpt_result['prediction']} | Actual: {actual} | Correct: {correct}")
    print("🧠 Confidence:", gpt_result["confidence"])
    print("📚 Reason:", gpt_result["reason"])

# ========== ENTRY POINT ==========
if __name__ == "__main__":
    run_daily_prediction()
