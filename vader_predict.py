import json
import os
import requests
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import random

# ========== CONFIGURATION ==========
NEWSAPI_KEY = ""  # <-- Insert your NewsAPI key here
LOG_FILE = "btc_predictions_log.json"
NUM_HEADLINES = 10
SAME_THRESHOLD = 50  # USD

# ========== NEWS & SENTIMENT ==========
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

def calculate_sentiment_vader(headlines):
    """Calculate average VADER sentiment score from a list of headlines."""
    analyzer = SentimentIntensityAnalyzer()
    if not headlines:
        return 0.0
    scores = [analyzer.polarity_scores(h)["compound"] for h in headlines]
    return round(sum(scores) / len(scores), 3)

# ========== PRICE DATA ==========
def get_price_data():
    """Simulate BTC price for yesterday and today (replace with real API later)."""
    yesterday_price = round(random.uniform(29000, 31000), 2)
    today_price = yesterday_price + round(random.uniform(-500, 500), 2)
    return yesterday_price, today_price

# ========== PREDICTION ==========
def classify_prediction(sentiment, price_diff, threshold=SAME_THRESHOLD):
    """Predict UP, DOWN or SAME."""
    if abs(price_diff) < threshold:
        return "SAME"
    if sentiment > 0.3 and price_diff > 0:
        return "UP"
    elif sentiment < -0.3 and price_diff < 0:
        return "DOWN"
    else:
        return "SAME"

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
    today = str(datetime.utcnow().date())
    yesterday = str(datetime.utcnow().date() - timedelta(days=1))

    try:
        headlines = fetch_bitcoin_headlines_newsapi(NEWSAPI_KEY, NUM_HEADLINES)
        sentiment = calculate_sentiment_vader(headlines)
    except Exception as e:
        print("Error fetching or processing news:", e)
        return

    y_price, t_price = get_price_data()
    price_diff = t_price - y_price

    prediction = classify_prediction(sentiment, price_diff)
    actual = get_actual_movement(y_price, t_price)
    correct = prediction == actual

    entry = {
        "date": today,
        "sentiment_score": sentiment,
        "headlines": headlines,
        "yesterday_price": y_price,
        "today_price": t_price,
        "prediction": prediction,
        "actual": actual,
        "correct": correct
    }

    logs = load_log()
    logs.append(entry)
    save_log(logs)

    print(f"[{today}] Prediction: {prediction} | Actual: {actual} | Correct: {correct}")
    print("Sentiment Score:", sentiment)
    print("Top Headlines:")
    for h in headlines:
        print(" •", h)

# ========== ENTRY POINT ==========
if __name__ == "__main__":
    run_daily_prediction()
    