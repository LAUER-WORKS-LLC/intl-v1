"""
02_sentiment_analysis — Data Collector
Collects sentiment data from various sources (news, social media, etc.)
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import os
from dotenv import load_dotenv

try:
    import praw
    REDDIT_AVAILABLE = True
except ImportError:
    REDDIT_AVAILABLE = False

load_dotenv()

# API Keys (set via environment variables)
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "evKEUv2Kzywwm2dk6uv1eaS0gnChH0mT")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")

# Reddit API credentials (optional - works without auth for read-only)
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "INT-L-Sentiment-Analysis/1.0")


class NewsDataCollector:
    """Collects news articles mentioning stocks"""
    
    def __init__(self, api_key: str = POLYGON_API_KEY):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io/v2"
    
    def fetch_news(self, ticker: str, start_date: str, end_date: str, 
                   limit: int = 100) -> List[Dict]:
        """
        Fetch news articles for a ticker
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Maximum number of articles to fetch
        
        Returns:
            List of news article dictionaries
        """
        url = f"{self.base_url}/reference/news"
        params = {
            "ticker": ticker,
            "published_utc.gte": start_date,
            "published_utc.lte": end_date,
            "limit": limit,
            "apiKey": self.api_key
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "OK":
                return data.get("results", [])
            else:
                print(f"⚠ API error for {ticker}: {data.get('message', 'Unknown error')}")
                return []
        
        except Exception as e:
            print(f"❌ Error fetching news for {ticker}: {str(e)}")
            return []
    
    def fetch_batch_news(self, tickers: List[str], start_date: str, 
                        end_date: str, delay: float = 0.1) -> Dict[str, List[Dict]]:
        """Fetch news for multiple tickers with rate limiting"""
        results = {}
        for ticker in tickers:
            articles = self.fetch_news(ticker, start_date, end_date)
            results[ticker] = articles
            time.sleep(delay)  # Rate limiting
        
        return results


class SocialMediaCollector:
    """Collects social media mentions (placeholder for Twitter/X integration)"""
    
    def __init__(self, bearer_token: Optional[str] = None):
        self.bearer_token = bearer_token or TWITTER_BEARER_TOKEN
        # Note: Twitter API v2 requires authentication setup
        # This is a placeholder structure
    
    def fetch_mentions(self, ticker: str, start_date: str, end_date: str,
                      limit: int = 100) -> List[Dict]:
        """
        Fetch social media mentions for a ticker
        
        Note: Requires Twitter API v2 setup
        This is a placeholder for future implementation
        """
        if not self.bearer_token:
            print("⚠ Twitter Bearer Token not configured. Skipping social media collection.")
            return []
        
        # TODO: Implement Twitter API v2 integration
        # Example structure:
        # - Use tweepy or requests with Bearer Token
        # - Search for ticker mentions
        # - Filter by date range
        # - Return list of tweets/posts with text and metadata
        
        return []


class GenericNewsCollector:
    """Collects news from generic news APIs (NewsAPI, etc.)"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or NEWSAPI_KEY
        self.base_url = "https://newsapi.org/v2"
    
    def fetch_news(self, query: str, start_date: str, end_date: str,
                   limit: int = 100) -> List[Dict]:
        """
        Fetch news articles using NewsAPI
        
        Args:
            query: Search query (can include ticker)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Maximum number of articles
        """
        if not self.api_key:
            print("⚠ NewsAPI key not configured. Skipping generic news collection.")
            return []
        
        url = f"{self.base_url}/everything"
        params = {
            "q": query,
            "from": start_date,
            "to": end_date,
            "sortBy": "publishedAt",
            "pageSize": min(limit, 100),  # NewsAPI max is 100 per page
            "apiKey": self.api_key
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "ok":
                return data.get("articles", [])
            else:
                print(f"⚠ NewsAPI error: {data.get('message', 'Unknown error')}")
                return []
        
        except Exception as e:
            print(f"❌ Error fetching news: {str(e)}")
            return []


class RedditCollector:
    """Collects Reddit posts and comments mentioning stocks"""
    
    def __init__(self, client_id: Optional[str] = None, 
                 client_secret: Optional[str] = None,
                 user_agent: Optional[str] = None):
        self.reddit = None
        if not REDDIT_AVAILABLE:
            print("⚠ PRAW not installed. Install with: pip install praw")
            return
        
        # Reddit API now requires authentication (no more read-only public access)
        # You need to create a Reddit app at https://www.reddit.com/prefs/apps
        try:
            if client_id and client_secret:
                self.reddit = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent=user_agent or REDDIT_USER_AGENT
                )
                # Test the connection
                try:
                    # Try to access a subreddit to test read access
                    test_sub = self.reddit.subreddit("test")
                    list(test_sub.hot(limit=1))  # Test read access
                    # If we get here, authentication worked
                except Exception as e:
                    print(f"  ⚠ Reddit authentication issue: {e}")
                    print("     Please check your Reddit API credentials")
                    print("     See REDDIT_SETUP.md for instructions")
                    self.reddit = None
            else:
                print("  ⚠ Reddit API credentials not provided")
                print("     Reddit now requires authentication.")
                print("     Create a Reddit app at: https://www.reddit.com/prefs/apps")
                print("     Then set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env")
                self.reddit = None
        except Exception as e:
            print(f"  ⚠ Could not initialize Reddit API: {e}")
            self.reddit = None
    
    def search_subreddit(self, subreddit_name: str, query: str, 
                        limit: int = 100, sort_by: str = "hot") -> List[Dict]:
        """
        Search for posts in a specific subreddit
        
        Args:
            subreddit_name: Name of subreddit (e.g., "stocks", "investing")
            query: Search query (ticker symbol)
            limit: Maximum number of posts
            sort_by: "hot", "new", "top", "rising"
        
        Returns:
            List of post dictionaries
        """
        if not self.reddit:
            return []
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            posts = []
            
            # Search for posts containing the query
            search_results = subreddit.search(query, limit=limit, sort=sort_by)
            
            for post in search_results:
                post_data = {
                    "title": post.title,
                    "text": post.selftext,
                    "score": post.score,
                    "created_utc": datetime.fromtimestamp(post.created_utc),
                    "url": post.url,
                    "subreddit": subreddit_name,
                    "author": str(post.author) if post.author else "deleted",
                    "num_comments": post.num_comments,
                    "source": "reddit_post"
                }
                posts.append(post_data)
            
            return posts
        
        except Exception as e:
            print(f"⚠ Error searching Reddit subreddit {subreddit_name}: {e}")
            return []
    
    def get_post_comments(self, post_id: str, limit: int = 100) -> List[Dict]:
        """Get comments from a specific Reddit post"""
        if not self.reddit:
            return []
        
        try:
            submission = self.reddit.submission(id=post_id)
            submission.comments.replace_more(limit=0)  # Flatten comment tree
            
            comments = []
            for comment in submission.comments.list()[:limit]:
                if hasattr(comment, 'body') and comment.body:
                    comment_data = {
                        "text": comment.body,
                        "score": comment.score,
                        "created_utc": datetime.fromtimestamp(comment.created_utc),
                        "author": str(comment.author) if comment.author else "deleted",
                        "source": "reddit_comment"
                    }
                    comments.append(comment_data)
            
            return comments
        
        except Exception as e:
            print(f"⚠ Error fetching Reddit comments: {e}")
            return []
    
    def search_stock_mentions(self, ticker: str, subreddits: List[str] = None,
                              limit_per_sub: int = 50) -> List[Dict]:
        """
        Search for mentions of a stock ticker across multiple subreddits
        
        Args:
            ticker: Stock ticker symbol
            subreddits: List of subreddit names to search (default: popular finance subreddits)
            limit_per_sub: Max posts per subreddit
        
        Returns:
            List of posts/comments mentioning the ticker
        """
        if not self.reddit:
            return []
        
        if subreddits is None:
            subreddits = ["stocks", "investing", "StockMarket", "wallstreetbets", 
                         "SecurityAnalysis", "valueinvesting", "options"]
        
        all_posts = []
        
        for subreddit in subreddits:
            print(f"  Searching r/{subreddit} for {ticker}...")
            posts = self.search_subreddit(subreddit, f"${ticker} OR {ticker}", 
                                         limit=limit_per_sub)
            all_posts.extend(posts)
            time.sleep(0.5)  # Rate limiting - Reddit allows 60 requests per minute
        
        # Also search in post titles and text directly
        # Some mentions might not be in search results
        for subreddit in subreddits[:3]:  # Limit to avoid too many requests
            try:
                subreddit_obj = self.reddit.subreddit(subreddit)
                # Get recent hot posts and check for ticker mentions
                for post in subreddit_obj.hot(limit=50):
                    text_content = (post.title + " " + post.selftext).upper()
                    if f"${ticker}" in text_content or f" {ticker} " in text_content:
                        if post.id not in [p.get("id") for p in all_posts if "id" in p]:
                            post_data = {
                                "title": post.title,
                                "text": post.selftext,
                                "score": post.score,
                                "created_utc": datetime.fromtimestamp(post.created_utc),
                                "url": post.url,
                                "subreddit": subreddit,
                                "author": str(post.author) if post.author else "deleted",
                                "num_comments": post.num_comments,
                                "source": "reddit_post"
                            }
                            all_posts.append(post_data)
            except Exception as e:
                print(f"  ⚠ Error fetching from r/{subreddit}: {e}")
        
        return all_posts


def save_news_data(news_data: Dict[str, List[Dict]], output_path: str):
    """Save collected news data to parquet file"""
    all_articles = []
    
    for ticker, articles in news_data.items():
        for article in articles:
            article["ticker"] = ticker
            all_articles.append(article)
    
    if all_articles:
        df = pd.DataFrame(all_articles)
        df.to_parquet(output_path)
        print(f"✓ Saved {len(all_articles)} articles to {output_path}")
    else:
        print("⚠ No articles to save")

