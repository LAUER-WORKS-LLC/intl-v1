# Memory Management Guide

## What's This Memory Thing About?

The memory warnings you're seeing are about **RAM (Random Access Memory)** - your computer's short-term working memory, not disk storage. Think of it like your desk space while working:

- **RAM** = Your desk (temporary workspace)
- **Disk Storage** = Your filing cabinet (permanent storage)

When processing 117K+ records across 302 tickers with 50+ features each, Python loads all this data into RAM. At 87% usage (14GB), you're using most of your available RAM.

## Why Is Memory High?

1. **Large DataFrames**: Each ticker has ~449 days × many features = lots of data
2. **Feature Computation**: Creating advanced features (R², ER, etc.) creates intermediate DataFrames
3. **Batch Accumulation**: All batches are stored in memory before combining
4. **Python's Memory Model**: Python doesn't immediately free memory; it waits for garbage collection

## Solutions

### Quick Fix: Run the Memory Cleanup Script

```bash
python clear_memory.py
```

This will:
- Force garbage collection (clean up unused objects)
- Show memory statistics
- Clear pandas caches
- Identify memory-hungry processes

### Better Fixes:

1. **Reduce Batch Size** (in `interactive_analytics.py`, line 131):
   - Change `batch_size=20` to `batch_size=10` or `5`
   - Smaller batches = less memory per batch

2. **Process Fewer Tickers**:
   - Use "TEST MODE" or "SMALL TEST" when selecting exchanges
   - Process in smaller chunks

3. **Close Other Applications**:
   - Close Excel, browsers, etc. to free RAM
   - You need ~16GB+ RAM for 300+ tickers

4. **Restart Python/IDE**:
   - Sometimes Python holds onto memory
   - Restart your IDE/terminal to get a fresh start

## Memory Management in the Code

The code already has memory management:

- ✅ **Batch Processing**: Processes 20 tickers at a time
- ✅ **Garbage Collection**: Calls `gc.collect()` after each batch
- ✅ **Memory Monitoring**: Checks memory before each batch
- ✅ **Cleanup**: Deletes intermediate DataFrames

## What Happens When Memory Is Too High?

- **85-90%**: Warning, forces garbage collection
- **>90%**: Skips batch (safety measure)
- **>95%**: System may slow down or swap to disk (very slow!)

## Tips

1. **Monitor Memory**: The script shows memory % before each batch
2. **Use Memory Cleanup**: Run `clear_memory.py` between runs
3. **Restart Between Runs**: If processing multiple times, restart Python
4. **Reduce Scope**: Process fewer tickers if memory is tight

## Technical Details

- **Memory Location**: All in RAM (short-term), not on disk
- **Garbage Collection**: Python automatically frees memory, but we force it
- **DataFrame Memory**: Each DataFrame with 117K rows × 50 columns ≈ 100-200MB
- **Feature Computation**: Creates temporary DataFrames (2-3x memory temporarily)

