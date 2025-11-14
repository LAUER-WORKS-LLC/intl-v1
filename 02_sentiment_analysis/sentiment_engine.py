"""
02_sentiment_analysis — Sentiment Analysis Engine
Part 2 of 4-part INT-L Analytics Series

Core engine for analyzing sentiment about stocks from various data sources.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import re

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class SentimentConfig:
    """Configuration for sentiment analysis"""
    analyzer_type: str = "vader"  # "vader", "textblob", "transformer", or "hybrid"
    min_confidence: float = 0.1
    use_weighted_scores: bool = True
    transformer_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"  # Twitter-RoBERTa (3-class: negative/neutral/positive)


class SentimentAnalyzer:
    """Sentiment analyzer for stock-related text"""
    
    def __init__(self, config: Optional[SentimentConfig] = None):
        self.config = config or SentimentConfig()
        
        # Initialize analyzers
        if VADER_AVAILABLE:
            self.vader_analyzer = SentimentIntensityAnalyzer()
        else:
            self.vader_analyzer = None
        
        # Initialize transformer-based NLP model (if available and requested)
        self.transformer_pipeline = None
        if self.config.analyzer_type in ["transformer", "hybrid"]:
            if not TRANSFORMERS_AVAILABLE:
                print("⚠ Transformers library not installed!")
                print("   Install with: pip install transformers torch")
                print("   Falling back to VADER/TextBlob")
                if self.config.analyzer_type == "transformer":
                    self.config.analyzer_type = "vader"  # Fallback
            else:
                try:
                    print("Loading transformer model (this may take a moment on first run)...")
                    self.transformer_pipeline = pipeline(
                        "sentiment-analysis",
                        model=self.config.transformer_model,
                        tokenizer=self.config.transformer_model,
                        device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
                        return_all_scores=False  # Return only top result
                    )
                    # Test the pipeline with a simple example
                    test_result = self.transformer_pipeline("This is great!")
                    if isinstance(test_result, list) and len(test_result) > 0:
                        test_result = test_result[0]
                    print(f"✓ Transformer model loaded")
                    print(f"  Test result: {test_result}")
                except Exception as e:
                    print(f"⚠ Could not load transformer model: {e}")
                    print("   Falling back to VADER/TextBlob")
                    import traceback
                    traceback.print_exc()
                    self.transformer_pipeline = None
                    if self.config.analyzer_type == "transformer":
                        self.config.analyzer_type = "vader"  # Fallback
            
        if not VADER_AVAILABLE and not TEXTBLOB_AVAILABLE and not TRANSFORMERS_AVAILABLE:
            raise ImportError("At least one sentiment analyzer must be installed (vaderSentiment, textblob, or transformers)")
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text
        
        Returns:
            Dictionary with sentiment scores:
            - sentiment_score: -1 (negative) to 1 (positive)
            - confidence: 0 to 1
            - compound: VADER compound score if available
            - polarity: TextBlob polarity if available
        """
        if not text or not text.strip():
            return {
                "sentiment_score": 0.0,
                "confidence": 0.0,
                "compound": 0.0,
                "polarity": 0.0
            }
        
        scores = {
            "sentiment_score": 0.0,
            "confidence": 0.0,
            "compound": 0.0,
            "polarity": 0.0
        }
        
        # VADER analysis (good for social media/short text)
        if self.vader_analyzer and self.config.analyzer_type in ["vader", "hybrid"]:
            vader_scores = self.vader_analyzer.polarity_scores(text)
            scores["compound"] = vader_scores["compound"]
            scores["confidence"] = abs(vader_scores["compound"])
            
            if self.config.analyzer_type == "vader":
                scores["sentiment_score"] = vader_scores["compound"]
        
        # TextBlob analysis (good for longer text)
        if TEXTBLOB_AVAILABLE and self.config.analyzer_type in ["textblob", "hybrid"]:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            scores["polarity"] = polarity
            
            if self.config.analyzer_type == "textblob":
                scores["sentiment_score"] = polarity
                scores["confidence"] = abs(polarity)
        
        # Transformer-based NLP analysis (most accurate, uses BERT/RoBERTa)
        transformer_score = 0.0
        transformer_confidence = 0.0
        
        # Check if transformer should be used
        if self.config.analyzer_type in ["transformer", "hybrid"]:
            if not self.transformer_pipeline:
                # Transformer not available - this shouldn't happen if selected, but handle gracefully
                if self.config.analyzer_type == "transformer":
                    print("⚠ Transformer not initialized! Falling back to VADER.")
                    # Fallback to VADER
                    if self.vader_analyzer:
                        vader_scores = self.vader_analyzer.polarity_scores(text)
                        scores["sentiment_score"] = vader_scores["compound"]
                        scores["confidence"] = abs(vader_scores["compound"])
                    return scores
                # For hybrid, continue with other analyzers
                return scores
            
        if self.transformer_pipeline and self.config.analyzer_type in ["transformer", "hybrid"]:
            try:
                # Limit text length to model's max (typically 512 tokens)
                # The model expects strings, not empty text
                if not text or not text.strip():
                    if self.config.analyzer_type == "transformer":
                        return scores
                    else:
                        pass  # Continue to hybrid
                else:
                    # Properly truncate to model's max length (512 tokens)
                    # Use tokenizer to truncate properly
                    if hasattr(self.transformer_pipeline, 'tokenizer'):
                        # Tokenize and truncate properly
                        tokens = self.transformer_pipeline.tokenizer.encode(
                            text, 
                            max_length=512, 
                            truncation=True,
                            return_tensors=None
                        )
                        # Decode back to text (this ensures proper truncation)
                        text_truncated = self.transformer_pipeline.tokenizer.decode(
                            tokens, 
                            skip_special_tokens=True
                        )
                    else:
                        # Fallback: character-based truncation (less ideal)
                        text_truncated = text[:500]  # Conservative truncation
                    
                    result = self.transformer_pipeline(text_truncated)
                    
                    # Handle both single result and list results
                    if isinstance(result, list):
                        if len(result) > 0:
                            result = result[0]
                        else:
                            # Empty result, skip
                            if self.config.analyzer_type == "transformer":
                                return scores
                    
                    # Debug: Print first result to see format
                    if not hasattr(self, '_transformer_debugged'):
                        print(f"\n  🔍 Transformer debug - First result format: {result}")
                        print(f"     Result type: {type(result)}")
                        if isinstance(result, dict):
                            print(f"     Result keys: {result.keys()}")
                            print(f"     Label: {result.get('label')}, Score: {result.get('score')}")
                        self._transformer_debugged = True
                    
                    # Convert label to score
                    # Handle different result formats
                    if isinstance(result, dict):
                        label = result.get("label", "")
                        confidence = result.get("score", 0.0)
                    elif isinstance(result, tuple):
                        # Some models return (label, score) tuple
                        label, confidence = result[0], result[1]
                    else:
                        # Fallback
                        label = str(result)
                        confidence = 0.5
                    
                    label_lower = str(label).lower()
                    
                    # Map labels to sentiment scores
                    # Different models use different label formats
                    # Twitter-RoBERTa uses: LABEL_0 (negative), LABEL_1 (neutral), LABEL_2 (positive)
                    # Other models use: "POSITIVE", "NEGATIVE", "NEUTRAL"
                    
                    if "positive" in label_lower or "pos" in label_lower or "LABEL_2" in str(label):
                        transformer_score = confidence
                    elif "negative" in label_lower or "neg" in label_lower or "LABEL_0" in str(label):
                        transformer_score = -confidence
                    elif "neutral" in label_lower or "neu" in label_lower or "LABEL_1" in str(label):
                        # For neutral, return small score based on confidence
                        transformer_score = 0.0
                    else:
                        # If we can't identify the label, check if it's a 3-class model
                        # Some models return all scores
                        if isinstance(result, dict) and "scores" in result:
                            # Multi-class result
                            scores_dict = result["scores"]
                            pos_score = scores_dict.get("POSITIVE", scores_dict.get("LABEL_2", 0))
                            neg_score = scores_dict.get("NEGATIVE", scores_dict.get("LABEL_0", 0))
                            transformer_score = pos_score - neg_score
                            confidence = max(pos_score, neg_score)
                        else:
                            # Default: assume high confidence means positive sentiment
                            transformer_score = confidence if confidence > 0.5 else -confidence
                    
                    transformer_confidence = confidence
                    scores["transformer"] = transformer_score
                    
                    if self.config.analyzer_type == "transformer":
                        scores["sentiment_score"] = transformer_score
                        scores["confidence"] = transformer_confidence
            except Exception as e:
                print(f"⚠ Transformer analysis error: {e}")
                import traceback
                traceback.print_exc()
                # Fallback: don't break, just log the error
        
        # Hybrid: combine multiple analyzers
        if self.config.analyzer_type == "hybrid":
            components = []
            weights = []
            
            if self.vader_analyzer and scores["compound"] != 0:
                components.append(scores["compound"])
                weights.append(0.3)  # VADER weight
            
            if TEXTBLOB_AVAILABLE and scores["polarity"] != 0:
                components.append(scores["polarity"])
                weights.append(0.2)  # TextBlob weight
            
            if self.transformer_pipeline and transformer_score != 0:
                components.append(transformer_score)
                weights.append(0.5)  # Transformer gets highest weight
            
            if components:
                # Normalize weights
                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]
                scores["sentiment_score"] = sum(c * w for c, w in zip(components, weights))
                scores["confidence"] = max([abs(scores.get("compound", 0)), 
                                           abs(scores.get("polarity", 0)), 
                                           transformer_confidence])
            else:
                # Fallback if no components available
                scores["sentiment_score"] = scores.get("compound", scores.get("polarity", 0))
        
        return scores
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Analyze sentiment for multiple texts"""
        return [self.analyze_text(text) for text in texts]


class SentimentAggregator:
    """Aggregates sentiment scores by ticker"""
    
    def __init__(self):
        self.by_ticker = defaultdict(list)
    
    def add_sentiment(self, ticker: str, sentiment_data: Dict[str, float], 
                     source: str = "unknown", timestamp: Optional[datetime] = None):
        """Add a sentiment score for a ticker"""
        entry = {
            "ticker": ticker,
            "sentiment_score": sentiment_data["sentiment_score"],
            "confidence": sentiment_data["confidence"],
            "source": source,
            "timestamp": timestamp or datetime.now()
        }
        self.by_ticker[ticker].append(entry)
    
    def get_ticker_metrics(self, ticker: str) -> Dict[str, float]:
        """Get aggregated metrics for a ticker"""
        if ticker not in self.by_ticker or not self.by_ticker[ticker]:
            return {
                "avg_sentiment": 0.0,
                "mention_count": 0,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "confidence": 0.0,
                "sentiment_score": 0.0
            }
        
        sentiments = self.by_ticker[ticker]
        scores = [s["sentiment_score"] for s in sentiments]
        confidences = [s["confidence"] for s in sentiments]
        
        # Weighted average by confidence
        if any(c > 0 for c in confidences):
            weighted_sum = sum(s * c for s, c in zip(scores, confidences))
            weight_sum = sum(confidences)
            avg_sentiment = weighted_sum / weight_sum
        else:
            avg_sentiment = np.mean(scores) if scores else 0.0
        
        positive_count = sum(1 for s in scores if s > 0.1)
        negative_count = sum(1 for s in scores if s < -0.1)
        
        return {
            "avg_sentiment": avg_sentiment,
            "mention_count": len(sentiments),
            "positive_ratio": positive_count / len(sentiments) if sentiments else 0.0,
            "negative_ratio": negative_count / len(sentiments) if sentiments else 0.0,
            "confidence": np.mean(confidences) if confidences else 0.0,
            "sentiment_score": avg_sentiment  # For consistency with Part 1
        }
    
    def get_all_ticker_metrics(self) -> pd.DataFrame:
        """Get metrics for all tickers"""
        results = []
        for ticker in self.by_ticker.keys():
            metrics = self.get_ticker_metrics(ticker)
            metrics["ticker"] = ticker
            results.append(metrics)
        
        df = pd.DataFrame(results)
        if len(df) > 0:
            df = df.sort_values("sentiment_score", ascending=False)
        return df


class StockMentionExtractor:
    """Extracts stock ticker mentions from text"""
    
    def __init__(self, ticker_list: List[str]):
        # Create regex pattern for tickers (case-insensitive, word boundaries)
        ticker_patterns = [rf'\b{re.escape(ticker)}\b' for ticker in ticker_list]
        self.ticker_pattern = re.compile('|'.join(ticker_patterns), re.IGNORECASE)
        self.ticker_list = [ticker.upper() for ticker in ticker_list]
    
    def extract_mentions(self, text: str) -> List[str]:
        """Extract mentioned tickers from text"""
        if not text:
            return []
        
        text_upper = text.upper()
        mentioned = []
        
        for ticker in self.ticker_list:
            if re.search(rf'\b{re.escape(ticker)}\b', text_upper):
                mentioned.append(ticker)
        
        return list(set(mentioned))  # Remove duplicates
    
    def extract_with_context(self, text: str, context_window: int = 50) -> Dict[str, List[str]]:
        """Extract tickers with surrounding context"""
        mentions = self.extract_mentions(text)
        result = {}
        
        for ticker in mentions:
            # Find all occurrences of ticker in text
            pattern = re.compile(rf'\b{re.escape(ticker)}\b', re.IGNORECASE)
            matches = pattern.finditer(text)
            
            contexts = []
            for match in matches:
                start = max(0, match.start() - context_window)
                end = min(len(text), match.end() + context_window)
                contexts.append(text[start:end].strip())
            
            result[ticker] = contexts
        
        return result

