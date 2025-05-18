import pandas as pd
import requests
import os
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pickle
import logging

# Configure logging
logging.basicConfig(
    filename='sentiment.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def fetch_news(ticker, days=30, api_key='769BKIGIFR49TEYZ'):
    cache_dir = 'sentiment_cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f'{ticker}_news.pkl')
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                news_df = pickle.load(f)
            news_df['time_published'] = pd.to_datetime(news_df['time_published'], errors='coerce')
            news_df = news_df[(news_df['time_published'] >= start_date) & 
                            (news_df['time_published'] <= end_date)]
            if not news_df.empty and not news_df['time_published'].isna().all():
                logging.info(f"Loaded cached news for {ticker} with {len(news_df)} articles")
                return news_df
            else:
                logging.warning(f"Empty or invalid cached news for {ticker}")
        except Exception as e:
            logging.error(f"Error loading cache for {ticker}: {e}")
    
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&time_from={start_date.strftime("%Y%m%dT%H%M")}&time_to={end_date.strftime("%Y%m%dT%H%M")}&apikey={api_key}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if 'feed' not in data or not data['feed']:
            logging.warning(f"No news data from API for {ticker}")
            return pd.DataFrame()
        
        news_data = []
        for article in data['feed']:
            time_published = pd.to_datetime(article.get('time_published', ''), errors='coerce')
            if pd.isna(time_published):
                continue
            news_data.append({
                'title': article.get('title', ''),
                'summary': article.get('summary', ''),
                'time_published': time_published,
                'ticker': ticker
            })
        
        news_df = pd.DataFrame(news_data)
        if news_df.empty:
            logging.warning(f"No valid news articles for {ticker}")
        else:
            logging.info(f"Fetched {len(news_df)} news articles for {ticker}")
            with open(cache_file, 'wb') as f:
                pickle.dump(news_df, f)
        return news_df
    except Exception as e:
        logging.error(f"Error fetching news for {ticker}: {e}")
        return pd.DataFrame()

def analyze_sentiment(news_df):
    if news_df.empty:
        logging.warning("Empty news DataFrame for sentiment analysis")
        return news_df
    
    analyzer = SentimentIntensityAnalyzer()
    
    def get_compound_score(text):
        if not text or pd.isna(text):
            return 0.0
        return analyzer.polarity_scores(text)['compound']
    
    news_df['title_sentiment'] = news_df['title'].apply(get_compound_score)
    news_df['summary_sentiment'] = news_df['summary'].apply(get_compound_score)
    news_df['sentiment'] = (news_df['title_sentiment'] + news_df['summary_sentiment']) / 2
    logging.info(f"Sentiment analysis completed, mean sentiment: {news_df['sentiment'].mean()}")
    return news_df

def aggregate_daily_sentiment(news_df):
    if news_df.empty:
        logging.warning("Empty news DataFrame for sentiment aggregation")
        return 0.0
    
    # Compute single 30-day sentiment score
    sentiment_score = news_df['sentiment'].mean()
    logging.info(f"Aggregated 30-day sentiment, score: {sentiment_score}")
    return sentiment_score