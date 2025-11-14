"""
02_sentiment_analysis — Part 2: Interactive Sentiment Analysis
Interactive sentiment analysis for stock ranking
Part 2 of 4-part INT-L Analytics Series
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
from typing import List, Dict, Optional, Tuple

from sentiment_engine import (
    SentimentAnalyzer, SentimentAggregator, StockMentionExtractor,
    SentimentConfig
)
from data_collector import NewsDataCollector, GenericNewsCollector, RedditCollector


# =====================================
# CONFIGURATION
# =====================================

# Exchange definitions (matching Part 1)
EXCHANGES = {
    "NYSE": "XNYS",
    "NASDAQ": "XNAS", 
    "NYSE_AMERICAN": "XASE",
    "NYSE_ARCA": "ARCX",
    "IEX": "IEX",
    "OTC": "OTC"
}

def load_stock_lists():
    """Load stock lists from Part 1's comprehensive list"""
    try:
        # Load from Part 1's stock list file
        import sys
        import os
        # Add parent directory to path to access Part 1 files
        part1_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '01_price_analysis')
        sys.path.insert(0, part1_path)
        from all_stocks_polygon_working_20251016_105954 import ALL_STOCKS, STOCKS_BY_EXCHANGE
        return ALL_STOCKS, STOCKS_BY_EXCHANGE
    except ImportError:
        # Fallback to a smaller default list if files not found
        default_stocks = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "BRK.A", "BRK.B", "UNH",
            "JNJ", "JPM", "V", "PG", "HD", "MA", "DIS", "PYPL", "ADBE", "NFLX",
            "CRM", "INTC", "AMD", "ABT", "TMO", "COST", "PFE", "PEP", "ABBV", "MRK",
            "ACN", "TXN", "AVGO", "DHR", "VZ", "NKE", "ADP", "QCOM", "NEE", "T",
            "LIN", "PM", "RTX", "HON", "SPGI", "LOW", "UNP", "UPS", "IBM", "GE"
        ]
        return default_stocks, {"DEFAULT": default_stocks}

def choose_exchanges():
    """Interactive exchange selection (same as Part 1)"""
    print("\n🏛️  EXCHANGE SELECTION")
    print("=" * 30)
    print("Available exchanges:")
    
    all_stocks, stocks_by_exchange = load_stock_lists()
    
    exchange_options = []
    for i, (exchange_name, exchange_code) in enumerate(EXCHANGES.items(), 1):
        stock_count = len(stocks_by_exchange.get(exchange_name, []))
        print(f"   {i}. {exchange_name} ({exchange_code}) - {stock_count} stocks")
        exchange_options.append((exchange_name, exchange_code))
    
    print(f"   {len(EXCHANGES) + 1}. ALL EXCHANGES - {len(all_stocks)} stocks")
    print(f"   {len(EXCHANGES) + 2}. CUSTOM LIST")
    print(f"   {len(EXCHANGES) + 3}. TEST MODE (first 50 stocks)")
    print(f"   {len(EXCHANGES) + 4}. SMALL TEST (first 10 stocks)")
    
    while True:
        try:
            choice = input(f"\nSelect exchanges (comma-separated numbers, 1-{len(EXCHANGES) + 4}): ").strip()
            
            if not choice:
                print("❌ Please make a selection")
                continue
                
            choices = [int(x.strip()) for x in choice.split(",") if x.strip().isdigit()]
            
            if len(EXCHANGES) + 1 in choices:  # ALL EXCHANGES
                return all_stocks, "ALL EXCHANGES"
            
            if len(EXCHANGES) + 2 in choices:  # CUSTOM LIST
                custom_tickers = input("Enter custom tickers (comma-separated): ").strip().split(",")
                custom_tickers = [t.strip().upper() for t in custom_tickers if t.strip()]
                return custom_tickers, "CUSTOM"
            
            if len(EXCHANGES) + 3 in choices:  # TEST MODE
                test_tickers = all_stocks[:50]
                return test_tickers, "TEST MODE (50 stocks)"
            
            if len(EXCHANGES) + 4 in choices:  # SMALL TEST
                test_tickers = all_stocks[:10]
                return test_tickers, "SMALL TEST (10 stocks)"
            
            selected_tickers = set()
            selected_exchanges = []
            
            for choice_num in choices:
                if 1 <= choice_num <= len(EXCHANGES):
                    exchange_name, exchange_code = exchange_options[choice_num - 1]
                    exchange_tickers = stocks_by_exchange.get(exchange_name, [])
                    selected_tickers.update(exchange_tickers)
                    selected_exchanges.append(exchange_name)
            
            if selected_tickers:
                return list(selected_tickers), f"SELECTED: {', '.join(selected_exchanges)}"
            else:
                print("❌ No valid exchanges selected")
                
        except ValueError:
            print("❌ Invalid input. Please enter numbers separated by commas.")
        except Exception as e:
            print(f"❌ Error: {e}")


def choose_date_range():
    """Interactive date range selection"""
    print("\n📅 DATE RANGE SELECTION")
    print("=" * 30)
    print("Date format: YYYY-MM-DD (e.g., 2024-01-01)")
    print("Leave blank for defaults")
    
    while True:
        start_date = input("Start date (default: 7 days ago): ").strip()
        if not start_date:
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        
        end_date = input("End date (default: today): ").strip()
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Validate date format
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
            datetime.strptime(end_date, "%Y-%m-%d")
            
            if start_date >= end_date:
                print("❌ Start date must be before end date")
                continue
            
            print(f"✓ Date range: {start_date} to {end_date}")
            return start_date, end_date
        
        except ValueError:
            print("❌ Invalid date format. Please use YYYY-MM-DD")


def choose_analyzer_type():
    """Choose sentiment analyzer type"""
    print("\n🔍 SENTIMENT ANALYZER SELECTION")
    print("=" * 30)
    print("1. VADER (rule-based, fast, good for social media)")
    print("2. TextBlob (pattern-based, good for longer articles)")
    print("3. Transformer NLP (BERT-based, most accurate, slower)")
    print("4. Hybrid (combines all available methods)")
    
    choice = input("Select analyzer (1-4, default: 1): ").strip()
    
    mapper = {
        "1": "vader",
        "2": "textblob",
        "3": "transformer",
        "4": "hybrid"
    }
    
    analyzer = mapper.get(choice, "vader")
    
    if analyzer == "transformer":
        print("\n⚠ Note: Transformer models require more memory and time.")
        print("   First run will download the model (~500MB).")
        print("   GPU acceleration recommended for faster processing.")
    
    return analyzer


def choose_data_sources():
    """Choose data sources for sentiment analysis"""
    print("\n📰 DATA SOURCE SELECTION")
    print("=" * 30)
    print("Select data sources (comma-separated numbers):")
    print("1. Polygon.io News (requires API key)")
    print("2. Generic News API (NewsAPI - requires API key)")
    print("3. Reddit (public API, works without auth)")
    print("4. Social Media / Twitter (placeholder - not yet implemented)")
    
    choices = input("Select sources (comma-separated, default: 1,3): ").strip()
    
    if not choices:
        return [1, 3]  # Default: Polygon + Reddit
    
    try:
        selected = [int(x.strip()) for x in choices.split(",")]
        return selected
    except ValueError:
        return [1, 3]


def collect_and_analyze_sentiment(tickers: List[str], start_date: str, 
                                   end_date: str, analyzer_type: str,
                                   data_sources: List[int]) -> Tuple[pd.DataFrame, SentimentAggregator]:
    """Main sentiment collection and analysis pipeline"""
    
    print(f"\n📥 COLLECTING DATA...")
    print("=" * 50)
    
    # Initialize collectors
    news_collector = NewsDataCollector()
    generic_collector = GenericNewsCollector()
    
    # Initialize Reddit collector with credentials if available
    import os
    from dotenv import load_dotenv
    load_dotenv()
    reddit_client_id = os.getenv("REDDIT_CLIENT_ID", "")
    reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
    reddit_collector = RedditCollector(
        client_id=reddit_client_id if reddit_client_id else None,
        client_secret=reddit_client_secret if reddit_client_secret else None
    )
    
    # Initialize sentiment components
    config = SentimentConfig(analyzer_type=analyzer_type)
    analyzer = SentimentAnalyzer(config)
    aggregator = SentimentAggregator()
    mention_extractor = StockMentionExtractor(tickers)
    
    # Collect news data
    all_articles = []
    
    if 1 in data_sources:  # Polygon.io News
        print("Fetching news from Polygon.io...")
        news_data = news_collector.fetch_batch_news(tickers, start_date, end_date)
        
        for ticker, articles in news_data.items():
            print(f"  {ticker}: {len(articles)} articles")
            for article in articles:
                article_text = article.get("title", "") + " " + article.get("description", "")
                if article_text.strip():
                    all_articles.append({
                        "ticker": ticker,
                        "text": article_text,
                        "source": "polygon_news",
                        "published": article.get("published_utc", "")
                    })
    
    if 2 in data_sources:  # Generic News API
        print("Fetching news from NewsAPI...")
        for ticker in tickers[:10]:  # Limit to avoid rate limits
            articles = generic_collector.fetch_news(ticker, start_date, end_date)
            for article in articles:
                article_text = article.get("title", "") + " " + article.get("description", "")
                if article_text.strip():
                    all_articles.append({
                        "ticker": ticker,
                        "text": article_text,
                        "source": "newsapi",
                        "published": article.get("publishedAt", "")
                    })
    
    if 3 in data_sources:  # Reddit
        print("Fetching posts from Reddit...")
        print("  (Searching popular finance subreddits: stocks, investing, wallstreetbets, etc.)")
        
        # Note: Reddit API has rate limits (60 requests/minute)
        # For large ticker lists, consider batching or limiting
        # We'll process all tickers but with rate limiting built in
        reddit_tickers = tickers  # Process all tickers (no artificial limit)
        
        # Show progress for large lists
        total_tickers = len(reddit_tickers)
        if total_tickers > 20:
            print(f"  Processing {total_tickers} tickers (this may take a while due to rate limits)...")
        
        for idx, ticker in enumerate(reddit_tickers, 1):
            if idx % 10 == 0 or idx == 1:
                print(f"  Searching Reddit for {ticker}... ({idx}/{total_tickers})")
            else:
                print(f"  Searching Reddit for {ticker}...", end='\r')
            
            reddit_posts = reddit_collector.search_stock_mentions(
                ticker, 
                subreddits=["stocks", "investing", "StockMarket", "wallstreetbets"],
                limit_per_sub=25
            )
            
            for post in reddit_posts:
                # Combine title and text for sentiment analysis
                post_text = post.get("title", "") + " " + post.get("text", "")
                if post_text.strip():
                    all_articles.append({
                        "ticker": ticker,
                        "text": post_text,
                        "source": f"reddit_{post.get('subreddit', 'unknown')}",
                        "published": post.get("created_utc", "").strftime("%Y-%m-%d") if isinstance(post.get("created_utc"), datetime) else "",
                        "score": post.get("score", 0),
                        "num_comments": post.get("num_comments", 0)
                    })
            
            # Rate limiting for Reddit API (60 requests/minute = 1 per second)
            time.sleep(1)
        
        print(f"  ✓ Found {sum(1 for a in all_articles if 'reddit' in a.get('source', ''))} Reddit posts")
    
    print(f"\n✓ Collected {len(all_articles)} total articles/posts")
    
    # Analyze sentiment
    print(f"\n🧮 ANALYZING SENTIMENT...")
    print("=" * 50)
    print(f"Using analyzer: {analyzer_type}")
    
    processed_count = 0
    for article in all_articles:
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"  Processed {processed_count}/{len(all_articles)} articles...")
        ticker = article["ticker"]
        text = article["text"]
        
        # Extract additional mentions
        mentioned_tickers = mention_extractor.extract_mentions(text)
        
        # Analyze sentiment
        sentiment = analyzer.analyze_text(text)
        
        # Add to aggregator for primary ticker
        aggregator.add_sentiment(ticker, sentiment, source=article["source"])
        
        # Add to aggregator for mentioned tickers
        for mentioned_ticker in mentioned_tickers:
            if mentioned_ticker != ticker:
                aggregator.add_sentiment(mentioned_ticker, sentiment, 
                                       source=f"mentioned_in_{ticker}")
    
    # Get aggregated results
    print(f"\n📊 GENERATING METRICS...")
    results_df = aggregator.get_all_ticker_metrics()
    
    return results_df, aggregator


def interactive_sentiment_analysis():
    """Main interactive sentiment analysis pipeline"""
    print("📰 INT-L Sentiment Analysis — Part 2")
    print("=" * 50)
    
    # Exchange and ticker selection (same as Part 1)
    tickers, exchange_description = choose_exchanges()
    print(f"\n✓ Selected {len(tickers)} tickers from {exchange_description}")
    
    # Interactive configuration
    start_date, end_date = choose_date_range()
    analyzer_type = choose_analyzer_type()
    data_sources = choose_data_sources()
    
    # Collect and analyze
    results_df, aggregator = collect_and_analyze_sentiment(
        tickers, start_date, end_date, analyzer_type, data_sources
    )
    
    # Display results
    print(f"\n{'='*20} SENTIMENT ANALYSIS RESULTS {'='*20}")
    print("\n📈 RANKED BY SENTIMENT SCORE:")
    print("-" * 80)
    print(f"{'Rank':<5} {'Ticker':<8} {'Sentiment':<12} {'Mentions':<10} {'Positive%':<12} {'Negative%':<12}")
    print("-" * 80)
    
    for i, row in results_df.iterrows():
        print(f"{i+1:<5} {row['ticker']:<8} {row['sentiment_score']:<12.3f} "
              f"{row['mention_count']:<10} {row['positive_ratio']*100:<12.1f}% "
              f"{row['negative_ratio']*100:<12.1f}%")
    
    # Show top/bottom performers
    print(f"\n🏆 HIGHEST REGARDED STOCKS:")
    top_5 = results_df.head(5)
    for i, row in top_5.iterrows():
        print(f"  {row['ticker']}: {row['sentiment_score']:.3f} ({row['mention_count']} mentions)")
    
    print(f"\n📉 LOWEST REGARDED STOCKS:")
    bottom_5 = results_df.tail(5)
    for i, row in bottom_5.iterrows():
        print(f"  {row['ticker']}: {row['sentiment_score']:.3f} ({row['mention_count']} mentions)")
    
    print(f"\n💬 MOST TALKED ABOUT:")
    most_mentioned = results_df.nlargest(5, 'mention_count')
    for i, row in most_mentioned.iterrows():
        print(f"  {row['ticker']}: {row['mention_count']} mentions (sentiment: {row['sentiment_score']:.3f})")
    
    # Save results
    save = input(f"\n💾 Save results to CSV? (y/n): ").lower()
    if save == "y":
        os.makedirs("results", exist_ok=True)
        output_path = f"results/sentiment_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(output_path, index=False)
        print(f"✓ Saved to {output_path}")
    
    print(f"\n✅ Sentiment Analysis Complete!")
    print(f"📊 Analyzed {len(results_df)} tickers")
    print(f"📅 Date range: {start_date} to {end_date}")


if __name__ == "__main__":
    interactive_sentiment_analysis()

