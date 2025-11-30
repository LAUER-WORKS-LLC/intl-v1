"""
Step 4: Analyze Chunk Coverage

This script:
1. For each ticker, calculates the total date span covered by all chunks
2. Checks for gaps greater than 5 days between consecutive chunks
3. Counts total number of chunks per ticker
4. Outputs results to terminal
"""

import os
import sys
import pandas as pd
from datetime import timedelta
from tqdm import tqdm
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def analyze_ticker_chunks(ticker, chunks_dir):
    """Analyze chunks for a single ticker"""
    ticker_chunks_dir = os.path.join(chunks_dir, ticker)
    
    if not os.path.exists(ticker_chunks_dir):
        return None
    
    # Get all chunk CSV files
    chunk_files = sorted([f for f in os.listdir(ticker_chunks_dir) if f.endswith('.csv')])
    
    if len(chunk_files) == 0:
        return None
    
    chunks_info = []
    
    # Read each chunk to get start/end dates
    for chunk_file in chunk_files:
        chunk_path = os.path.join(ticker_chunks_dir, chunk_file)
        try:
            chunk_data = pd.read_csv(chunk_path, index_col=0, parse_dates=True)
            
            if len(chunk_data) == 0:
                continue
            
            start_date = chunk_data.index[0]
            end_date = chunk_data.index[-1]
            
            chunks_info.append({
                'file': chunk_file,
                'start_date': start_date,
                'end_date': end_date,
                'duration': len(chunk_data)
            })
        except Exception as e:
            continue
    
    if len(chunks_info) == 0:
        return None
    
    # Sort by start date
    chunks_info.sort(key=lambda x: x['start_date'])
    
    # Calculate total span
    first_start = chunks_info[0]['start_date']
    last_end = chunks_info[-1]['end_date']
    total_span = (last_end - first_start).days
    
    # Check for gaps > 5 days between consecutive chunks
    gaps = []
    for i in range(len(chunks_info) - 1):
        current_end = chunks_info[i]['end_date']
        next_start = chunks_info[i + 1]['start_date']
        gap_days = (next_start - current_end).days
        
        if gap_days > 5:
            gaps.append({
                'after_chunk': chunks_info[i]['file'],
                'gap_days': gap_days,
                'from_date': current_end,
                'to_date': next_start
            })
    
    return {
        'ticker': ticker,
        'total_chunks': len(chunks_info),
        'first_chunk_start': first_start,
        'last_chunk_end': last_end,
        'total_span_days': total_span,
        'gaps_gt_5_days': len(gaps),
        'gap_details': gaps
    }

def main():
    print("="*80)
    print("STEP 4: ANALYZE CHUNK COVERAGE")
    print("="*80)
    
    chunks_dir = config.CHUNKS_DIR
    
    if not os.path.exists(chunks_dir):
        print(f"\n[ERROR] Chunks directory not found: {chunks_dir}")
        return
    
    # Get all ticker directories
    tickers = [d for d in os.listdir(chunks_dir) if os.path.isdir(os.path.join(chunks_dir, d))]
    tickers.sort()
    
    if len(tickers) == 0:
        print(f"\n[WARNING] No ticker directories found in {chunks_dir}")
        return
    
    print(f"\nAnalyzing {len(tickers)} tickers...\n")
    
    results = []
    
    for ticker in tqdm(tickers, desc="Analyzing"):
        analysis = analyze_ticker_chunks(ticker, chunks_dir)
        if analysis:
            results.append(analysis)
    
    # Sort results by ticker
    results.sort(key=lambda x: x['ticker'])
    
    # Output results
    print("\n" + "="*80)
    print("CHUNK COVERAGE ANALYSIS RESULTS")
    print("="*80)
    
    for result in results:
        print(f"\n{result['ticker']}:")
        print(f"  Total Chunks: {result['total_chunks']}")
        print(f"  Date Span: {result['first_chunk_start'].strftime('%Y-%m-%d')} to {result['last_chunk_end'].strftime('%Y-%m-%d')}")
        print(f"  Total Span: {result['total_span_days']} days")
        
        if result['gaps_gt_5_days'] > 0:
            print(f"  Gaps > 5 days: {result['gaps_gt_5_days']}")
            for gap in result['gap_details']:
                print(f"    - {gap['gap_days']} days gap after {gap['after_chunk']}")
                print(f"      From: {gap['from_date'].strftime('%Y-%m-%d')} to {gap['to_date'].strftime('%Y-%m-%d')}")
        else:
            print(f"  Gaps > 5 days: None")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    if len(results) > 0:
        total_chunks = sum(r['total_chunks'] for r in results)
        avg_chunks = total_chunks / len(results)
        tickers_with_gaps = sum(1 for r in results if r['gaps_gt_5_days'] > 0)
        avg_span = sum(r['total_span_days'] for r in results) / len(results)
        
        print(f"Total Tickers Analyzed: {len(results)}")
        print(f"Total Chunks: {total_chunks}")
        print(f"Average Chunks per Ticker: {avg_chunks:.1f}")
        print(f"Tickers with Gaps > 5 days: {tickers_with_gaps}")
        print(f"Average Date Span: {avg_span:.1f} days")
        
        # Show tickers with most chunks
        top_chunks = sorted(results, key=lambda x: x['total_chunks'], reverse=True)[:10]
        print(f"\nTop 10 Tickers by Chunk Count:")
        for r in top_chunks:
            print(f"  {r['ticker']}: {r['total_chunks']} chunks, {r['total_span_days']} days span")
        
        # Show tickers with gaps
        if tickers_with_gaps > 0:
            tickers_with_gaps_list = [r for r in results if r['gaps_gt_5_days'] > 0]
            print(f"\nTickers with Gaps > 5 days ({len(tickers_with_gaps_list)}):")
            for r in sorted(tickers_with_gaps_list, key=lambda x: x['gaps_gt_5_days'], reverse=True):
                print(f"  {r['ticker']}: {r['gaps_gt_5_days']} gap(s)")

if __name__ == "__main__":
    main()

