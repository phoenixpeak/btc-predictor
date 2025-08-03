from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests

def fetch_bitcoin_headlines_newsapi(api_key, num_articles=20):
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

# Example usage
if __name__ == "__main__":
    API_KEY = ""
    headlines = fetch_bitcoin_headlines_newsapi(API_KEY)
    sentiment_score = calculate_sentiment_vader(headlines)

    print("Top headlines:")
    for hl in headlines:
        print("•", hl)
    print("\nAverage sentiment score:", sentiment_score)
