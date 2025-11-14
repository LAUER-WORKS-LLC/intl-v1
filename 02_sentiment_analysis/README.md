# 02_sentiment_analysis — Part 2: Sentiment Analysis

**Part 2 of 4-part INT-L Analytics Series**

This module analyzes sentiment about stocks from news articles, social media, and other text sources. It identifies what stocks are being talked about, the general sentiment (positive/negative), and provides rankings like most talked about, highest-regarded, and lowest-regarded stocks.

## Overview

The sentiment analysis module:
- Collects text data about stocks from multiple sources (news APIs, social media)
- Analyzes sentiment using NLP techniques (VADER, TextBlob, or hybrid)
- Aggregates sentiment scores by ticker
- Ranks stocks by sentiment, mention count, and other metrics
- Provides insights: most talked about, highest-regarded, lowest-regarded stocks

## Features

- **Multi-Source Data Collection**: 
  - Polygon.io News API
  - NewsAPI (generic news)
  - Social Media (Twitter/X - placeholder for future implementation)
  
- **Sentiment Analysis**:
  - VADER (optimized for social media/short text)
  - TextBlob (good for longer articles)
  - Hybrid (combines both for better accuracy)
  
- **Stock Mention Extraction**: Automatically detects ticker mentions in text
  
- **Rankings & Metrics**:
  - Most talked about stocks
  - Highest-regarded stocks (positive sentiment)
  - Lowest-regarded stocks (negative sentiment)
  - Average sentiment scores
  - Mention counts and ratios

## Quick Start

### 1. Install Dependencies
```bash
cd 02_sentiment_analysis
pip install -r requirements.txt
```

### 2. Configure API Keys (Optional)
Create a `.env` file in the project root:
```bash
POLYGON_API_KEY=your_polygon_key_here  # Already configured by default
NEWSAPI_KEY=your_newsapi_key_here  # Optional
REDDIT_CLIENT_ID=your_reddit_client_id  # Optional (works without auth)
REDDIT_CLIENT_SECRET=your_reddit_client_secret  # Optional
REDDIT_USER_AGENT=INT-L-Sentiment-Analysis/1.0  # Optional
TWITTER_BEARER_TOKEN=your_twitter_token_here  # For future implementation
```

**Note:** Reddit works without authentication in read-only mode. Authentication provides higher rate limits.

### 3. Run Interactive Sentiment Analysis
```bash
python interactive_sentiment.py
```

The interactive system will guide you through:
1. **Date Range**: Select time period to analyze
2. **Analyzer Type**: Choose VADER, TextBlob, or Hybrid
3. **Data Sources**: Select which APIs to use
4. **Results**: View ranked stocks with sentiment metrics

## File Structure

```
02_sentiment_analysis/
├── interactive_sentiment.py      # Main entry point - interactive sentiment analysis
├── sentiment_engine.py            # Core sentiment analysis engine
├── data_collector.py              # Data collection from various sources
├── data/                          # Collected news/social media data (parquet)
├── results/                       # Sentiment analysis outputs (CSV)
├── README.md                      # Part 2 documentation
└── requirements.txt               # Python dependencies
```

## Core Components

### `sentiment_engine.py`
- `SentimentAnalyzer`: Analyzes text sentiment using VADER/TextBlob
- `SentimentAggregator`: Aggregates sentiment scores by ticker
- `StockMentionExtractor`: Extracts ticker mentions from text

### `data_collector.py`
- `NewsDataCollector`: Fetches news from Polygon.io
- `GenericNewsCollector`: Fetches news from NewsAPI
- `SocialMediaCollector`: Placeholder for Twitter/X integration

### `interactive_sentiment.py`
- Main interactive application
- Orchestrates data collection and sentiment analysis
- Displays results and rankings

## Sentiment Metrics

The analysis provides:

1. **Sentiment Score**: -1 (very negative) to +1 (very positive)
2. **Mention Count**: Number of times stock was mentioned
3. **Positive Ratio**: Percentage of positive mentions
4. **Negative Ratio**: Percentage of negative mentions
5. **Confidence**: Reliability of sentiment score
6. **Average Sentiment**: Weighted average sentiment (by confidence)

## Rankings Provided

1. **Ranked by Sentiment Score**: Overall sentiment ranking (best to worst)
2. **Most Talked About**: Stocks with highest mention counts
3. **Highest-Regarded**: Top 5 stocks by positive sentiment
4. **Lowest-Regarded**: Bottom 5 stocks by negative sentiment

## Data Sources

### Polygon.io News
- Stock-specific news articles
- Requires Polygon.io API key (already configured)
- Free tier available

### NewsAPI
- General news articles
- Requires NewsAPI key
- Free tier available (100 requests/day)

### Reddit
- Posts and comments from popular finance subreddits
- Searches: r/stocks, r/investing, r/StockMarket, r/wallstreetbets, etc.
- **Works without authentication** (read-only mode)
- Higher rate limits with Reddit API credentials (optional)
- Free to use

### Social Media (Future)
- Twitter/X mentions (placeholder)
- Requires Twitter API v2 access

## Integration with Other Parts

This module (Part 2) focuses on sentiment analysis. The output sentiment scores can be:
- Used standalone for sentiment-based stock selection
- Combined with Part 1 (Price Analysis) for multi-factor ranking
- Input for Part 3: (TBD)
- Input for Part 4: (TBD)

## Output

Results are saved to:
- Console: Formatted rankings and metrics
- CSV: `results/sentiment_scores_YYYYMMDD_HHMMSS.csv`

CSV columns:
- `ticker`: Stock symbol
- `sentiment_score`: Average sentiment (-1 to +1)
- `mention_count`: Number of mentions
- `positive_ratio`: Ratio of positive mentions
- `negative_ratio`: Ratio of negative mentions
- `confidence`: Average confidence score

## Requirements

See `requirements.txt` for full dependency list. Key dependencies:
- pandas, numpy: Data processing
- vaderSentiment: VADER sentiment analyzer
- textblob: TextBlob sentiment analyzer
- requests: API calls
- tweepy: Twitter API (for future implementation)
- python-dotenv: Environment variable management

## Notes

- **API Rate Limits**: Be aware of API rate limits when analyzing large ticker lists
- **Date Range**: Sentiment analysis works best with recent data (last 7-30 days)
- **Ticker List**: Defaults to loading from Part 1's stock list, with fallback to default list
- **Data Sources**: Configure API keys in `.env` file or environment variables

## Future Enhancements

- [ ] Twitter/X API integration for social media sentiment
- [ ] Reddit sentiment analysis
- [ ] Time-weighted sentiment (recent mentions weighted more)
- [ ] Industry-specific sentiment analysis
- [ ] Sentiment trend analysis over time
- [ ] Integration with Part 1 for combined rankings

