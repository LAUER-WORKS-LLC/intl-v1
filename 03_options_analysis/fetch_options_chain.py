"""
Options Chain Fetcher for RKLB
Fetches all call options with Greeks and market data using Polygon API or yfinance
"""

import pandas as pd
import requests
import json
from datetime import datetime, date
from typing import List, Dict, Optional
import os
import time
import numpy as np
from scipy.stats import norm
from math import log, sqrt, exp

# Polygon API Key (from 01_price_analysis/download_data.py)
API_KEY = "evKEUv2Kzywwm2dk6uv1eaS0gnChH0mT"


def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"):
    """
    Calculate option Greeks using Black-Scholes model
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free rate (assume 0.05 or 5% if not provided)
        sigma: Implied volatility (as decimal, e.g., 0.25 for 25%)
        option_type: "call" or "put"
    
    Returns:
        Dictionary with delta, gamma, theta, vega
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {"delta": None, "gamma": None, "theta": None, "vega": None}
    
    # Default risk-free rate if not provided
    if r is None or r <= 0:
        r = 0.05
    
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    
    # Delta
    if option_type.lower() == "call":
        delta = norm.cdf(d1)
    else:
        delta = -norm.cdf(-d1)
    
    # Gamma (same for calls and puts)
    gamma = norm.pdf(d1) / (S * sigma * sqrt(T))
    
    # Theta (per day, negative for time decay)
    if option_type.lower() == "call":
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * sqrt(T)) - r * K * exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * sqrt(T)) + r * K * exp(-r * T) * norm.cdf(-d2)) / 365
    
    # Vega (same for calls and puts, per 1% change in volatility)
    vega = S * norm.pdf(d1) * sqrt(T) / 100
    
    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega
    }


def fetch_options_chain_polygon(ticker: str) -> Optional[Dict]:
    """
    Fetch options chain snapshot from Polygon API
    
    Args:
        ticker: Stock ticker symbol (e.g., 'RKLB')
    
    Returns:
        Dictionary containing options chain data or None if error
    """
    url = f"https://api.polygon.io/v3/snapshot/options/{ticker}"
    params = {"apiKey": API_KEY}
    
    try:
        print(f"Fetching options chain for {ticker} from Polygon...")
        response = requests.get(url, params=params)
        
        # Check for 403 or other errors
        if response.status_code == 403:
            print("Warning: Polygon API returned 403 Forbidden. Your API key may not have options data access.")
            print("Falling back to yfinance...")
            return None
        
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") != "OK":
            print(f"Warning: API error: {data.get('message', 'Unknown error')}")
            return None
        
        return data
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            print("Warning: Polygon API access denied (403). Your subscription may not include options data.")
            print("Falling back to yfinance...")
        else:
            print(f"Warning: HTTP error fetching from Polygon: {str(e)}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Warning: Error fetching from Polygon: {str(e)}")
        return None
    except Exception as e:
        print(f"Warning: Unexpected error: {str(e)}")
        return None


def parse_options_chain(data: Dict, ticker: str) -> pd.DataFrame:
    """
    Parse Polygon options chain response and extract call options with Greeks
    
    Args:
        data: Polygon API response dictionary
        ticker: Stock ticker symbol
    
    Returns:
        DataFrame with call options and their metrics
    """
    options_list = []
    
    # Polygon v3 snapshot returns data in 'results' field
    results = data.get("results", [])
    
    if not results:
        print("Warning: No options data found in response")
        return pd.DataFrame()
    
    # Iterate through all options contracts
    for contract in results:
        # Check if it's a call option
        details = contract.get("details", {})
        contract_type = details.get("contract_type", "").upper()
        
        if contract_type != "CALL":
            continue
        
        # Extract option details
        option_data = {
            "ticker": ticker,
            "option_symbol": contract.get("ticker", ""),
            "expiration_date": details.get("expiration_date", ""),
            "strike_price": details.get("strike_price", None),
            "contract_type": contract_type,
        }
        
        # Extract Greeks from the contract
        greeks = contract.get("greeks", {})
        if greeks:
            option_data["delta"] = greeks.get("delta")
            option_data["gamma"] = greeks.get("gamma")
            option_data["theta"] = greeks.get("theta")
            option_data["vega"] = greeks.get("vega")  # Bonus metric
            option_data["implied_volatility"] = greeks.get("implied_volatility")
        
        # Extract market data
        last_quote = contract.get("last_quote", {})
        if last_quote:
            option_data["bid"] = last_quote.get("bid")
            option_data["ask"] = last_quote.get("ask")
            option_data["mid_price"] = (option_data.get("bid", 0) + option_data.get("ask", 0)) / 2 if option_data.get("bid") and option_data.get("ask") else None
        
        # Extract volume and open interest
        day = contract.get("day", {})
        if day:
            option_data["volume"] = day.get("volume")
            option_data["open_interest"] = day.get("open_interest")
            option_data["close_price"] = day.get("close")
            option_data["high"] = day.get("high")
            option_data["low"] = day.get("low")
        
        # Extract previous day data if available
        prev_day = contract.get("prev_day", {})
        if prev_day:
            option_data["prev_day_close"] = prev_day.get("close")
            option_data["prev_day_volume"] = prev_day.get("volume")
        
        options_list.append(option_data)
    
    if not options_list:
        print("Warning: No call options found in the chain")
        return pd.DataFrame()
    
    df = pd.DataFrame(options_list)
    
    # Sort by expiration date and strike price
    if "expiration_date" in df.columns and "strike_price" in df.columns:
        df["expiration_date"] = pd.to_datetime(df["expiration_date"], errors="coerce")
        df = df.sort_values(["expiration_date", "strike_price"])
    
    return df


def fetch_options_yfinance(ticker: str) -> pd.DataFrame:
    """
    Fetch options chain using yfinance and calculate Greeks if needed
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        DataFrame with call options including Greeks
    """
    try:
        import yfinance as yf
        print(f"Fetching options chain for {ticker} from yfinance...")
        
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get current stock price
        current_price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
        if not current_price:
            # Try to get from recent data
            hist = stock.history(period="1d")
            if not hist.empty:
                current_price = hist["Close"].iloc[-1]
            else:
                print("Warning: Could not determine current stock price for Greeks calculation")
                current_price = None
        
        expirations = stock.options
        
        if not expirations:
            print("Warning: No options expirations found via yfinance")
            return pd.DataFrame()
        
        print(f"Found {len(expirations)} expiration dates")
        all_options = []
        
        # Fetch options for each expiration
        for i, exp_date in enumerate(expirations):
            try:
                print(f"  Fetching expiration {i+1}/{len(expirations)}: {exp_date}...")
                opt_chain = stock.option_chain(exp_date)
                calls = opt_chain.calls.copy()
                
                # Add expiration date to each row
                calls["expiration_date"] = pd.to_datetime(exp_date)
                calls["ticker"] = ticker
                calls["contract_type"] = "CALL"
                
                # Rename columns to match our format
                column_mapping = {
                    "strike": "strike_price",
                    "lastPrice": "close_price",
                    "bid": "bid",
                    "ask": "ask",
                    "volume": "volume",
                    "openInterest": "open_interest",
                    "impliedVolatility": "implied_volatility",
                }
                
                # Only rename columns that exist
                for old_col, new_col in column_mapping.items():
                    if old_col in calls.columns:
                        calls = calls.rename(columns={old_col: new_col})
                
                # Calculate mid price
                if "bid" in calls.columns and "ask" in calls.columns:
                    calls["mid_price"] = (calls["bid"] + calls["ask"]) / 2
                elif "close_price" in calls.columns:
                    calls["mid_price"] = calls["close_price"]
                
                # Calculate time to expiration in years
                exp_datetime = pd.to_datetime(exp_date)
                today = pd.Timestamp.now()
                time_to_exp = (exp_datetime - today).days / 365.0
                calls["time_to_expiration"] = time_to_exp
                
                # Calculate Greeks if we have implied volatility and current price
                if current_price and "implied_volatility" in calls.columns and "strike_price" in calls.columns:
                    print(f"    Calculating Greeks for {len(calls)} contracts...")
                    
                    # Convert IV from percentage to decimal if needed
                    iv_col = calls["implied_volatility"]
                    if iv_col.max() > 1:  # Assume it's in percentage form
                        iv_decimal = iv_col / 100.0
                    else:
                        iv_decimal = iv_col
                    
                    # Calculate Greeks for each option
                    greeks_list = []
                    for idx, row in calls.iterrows():
                        greeks = calculate_greeks(
                            S=current_price,
                            K=row["strike_price"],
                            T=max(time_to_exp, 0.001),  # Avoid division by zero
                            r=0.05,  # Assume 5% risk-free rate
                            sigma=iv_decimal.iloc[calls.index.get_loc(idx)] if isinstance(iv_decimal, pd.Series) else iv_decimal,
                            option_type="call"
                        )
                        greeks_list.append(greeks)
                    
                    greeks_df = pd.DataFrame(greeks_list)
                    calls["delta"] = greeks_df["delta"]
                    calls["gamma"] = greeks_df["gamma"]
                    calls["theta"] = greeks_df["theta"]
                    calls["vega"] = greeks_df["vega"]
                
                all_options.append(calls)
                time.sleep(0.2)  # Rate limiting
                
            except Exception as e:
                print(f"  Warning: Error fetching expiration {exp_date}: {str(e)}")
                continue
        
        if not all_options:
            return pd.DataFrame()
        
        df = pd.concat(all_options, ignore_index=True)
        
        # Select and order the columns we want
        desired_columns = [
            "ticker", "expiration_date", "strike_price", "contract_type",
            "delta", "gamma", "theta", "implied_volatility", "volume", "open_interest",
            "bid", "ask", "mid_price", "close_price", "vega"
        ]
        
        # Only include columns that exist
        available_columns = [col for col in desired_columns if col in df.columns]
        df = df[available_columns]
        
        # Sort by expiration date and strike price
        if "expiration_date" in df.columns and "strike_price" in df.columns:
            df = df.sort_values(["expiration_date", "strike_price"]).reset_index(drop=True)
        
        return df
        
    except ImportError:
        print("Error: yfinance not available. Install with: pip install yfinance")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error: Error with yfinance: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def save_options_data(df: pd.DataFrame, ticker: str, output_dir: str = "."):
    """
    Save options data to CSV and JSON files
    
    Args:
        df: DataFrame with options data
        ticker: Stock ticker symbol
        output_dir: Output directory path
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as CSV
    csv_path = os.path.join(output_dir, f"{ticker}_options_chain_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Success: Saved CSV: {csv_path}")
    
    # Save as JSON
    json_path = os.path.join(output_dir, f"{ticker}_options_chain_{timestamp}.json")
    df.to_json(json_path, orient="records", indent=2, date_format="iso")
    print(f"Success: Saved JSON: {json_path}")
    
    # Save summary statistics
    summary_path = os.path.join(output_dir, f"{ticker}_options_summary_{timestamp}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Options Chain Summary for {ticker}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total Call Options: {len(df)}\n")
        f.write(f"Fetch Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if "expiration_date" in df.columns:
            f.write(f"Expiration Dates: {df['expiration_date'].nunique()} unique dates\n")
            f.write(f"Date Range: {df['expiration_date'].min()} to {df['expiration_date'].max()}\n\n")
        
        if "strike_price" in df.columns:
            f.write(f"Strike Price Range: ${df['strike_price'].min():.2f} to ${df['strike_price'].max():.2f}\n\n")
        
        # Statistics on Greeks
        greeks = ["delta", "gamma", "theta", "implied_volatility"]
        for greek in greeks:
            if greek in df.columns:
                greek_data = df[greek].dropna()
                if len(greek_data) > 0:
                    f.write(f"{greek.capitalize().replace('_', ' ')}:\n")
                    f.write(f"  Mean: {greek_data.mean():.6f}\n")
                    f.write(f"  Min: {greek_data.min():.6f}\n")
                    f.write(f"  Max: {greek_data.max():.6f}\n")
                    f.write(f"  Count (non-null): {len(greek_data)}/{len(df)}\n\n")
        
        # Volume and Open Interest stats
        if "volume" in df.columns:
            vol_data = df["volume"].dropna()
            if len(vol_data) > 0:
                f.write(f"Total Volume: {vol_data.sum():,.0f}\n")
                f.write(f"Average Volume per Contract: {vol_data.mean():.2f}\n")
                f.write(f"Contracts with Volume > 0: {(vol_data > 0).sum()}\n\n")
        
        if "open_interest" in df.columns:
            oi_data = df["open_interest"].dropna()
            if len(oi_data) > 0:
                f.write(f"Total Open Interest: {oi_data.sum():,.0f}\n")
                f.write(f"Average Open Interest per Contract: {oi_data.mean():.2f}\n")
                f.write(f"Contracts with OI > 0: {(oi_data > 0).sum()}\n")
    
    print(f"Success: Saved Summary: {summary_path}")


def main():
    """Main function to fetch and save RKLB options chain"""
    ticker = "RKLB"
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 60)
    print(f"Fetching Options Chain for {ticker}")
    print("=" * 60)
    print()
    
    # Try Polygon first (may require higher subscription tier)
    polygon_data = fetch_options_chain_polygon(ticker)
    
    if polygon_data:
        df = parse_options_chain(polygon_data, ticker)
        
        if not df.empty:
            print(f"\nSuccess: Fetched {len(df)} call options from Polygon")
            print(f"\nColumns available: {', '.join(df.columns)}")
            print(f"\nSample data:")
            print(df.head())
            
            # Save the data in the script's directory
            save_options_data(df, ticker, output_dir=script_dir)
            
            return df
        else:
            print("\nPolygon returned empty results, trying yfinance...")
    else:
        print("\nTrying yfinance...")
    
    # Use yfinance (more accessible, will calculate Greeks)
    df = fetch_options_yfinance(ticker)
    
    if not df.empty:
        print(f"\nSuccess: Fetched {len(df)} call options from yfinance")
        print(f"\nColumns available: {', '.join(df.columns)}")
        print(f"\nSample data (first 5 rows):")
        print(df.head())
        
        # Save the data in the script's directory
        save_options_data(df, ticker, output_dir=script_dir)
        
        return df
    else:
        print("\nError: Failed to fetch options chain from both sources")
        return pd.DataFrame()


if __name__ == "__main__":
    main()
