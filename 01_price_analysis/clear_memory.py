"""
Memory Cleanup Utility
Clears Python memory and provides memory diagnostics
"""

import gc
import sys
import psutil
import os

def clear_memory():
    """Aggressively clear Python memory"""
    print("🧹 Clearing Python memory...")
    
    # Force garbage collection multiple times
    for i in range(3):
        collected = gc.collect()
        print(f"   GC cycle {i+1}: Collected {collected} objects")
    
    # Clear any cached values
    gc.collect()
    
    # Get memory stats
    memory = psutil.virtual_memory()
    memory_mb = memory.used / 1024 / 1024
    memory_percent = memory.percent
    
    print(f"\n✓ Memory after cleanup:")
    print(f"   Used: {memory_mb:.1f} MB ({memory_percent:.1f}%)")
    print(f"   Available: {memory.available / 1024 / 1024:.1f} MB")
    
    return memory_percent

def show_memory_stats():
    """Show detailed memory statistics"""
    memory = psutil.virtual_memory()
    
    print("\n📊 MEMORY STATISTICS")
    print("=" * 50)
    print(f"Total RAM:     {memory.total / 1024 / 1024 / 1024:.2f} GB")
    print(f"Used:          {memory.used / 1024 / 1024 / 1024:.2f} GB ({memory.percent:.1f}%)")
    print(f"Available:    {memory.available / 1024 / 1024 / 1024:.2f} GB")
    print(f"Free:          {memory.free / 1024 / 1024 / 1024:.2f} GB")
    
    # Show process memory
    process = psutil.Process(os.getpid())
    process_memory = process.memory_info()
    print(f"\nPython Process:")
    print(f"RSS:           {process_memory.rss / 1024 / 1024:.1f} MB")
    print(f"VMS:           {process_memory.vms / 1024 / 1024:.1f} MB")
    
    # Show top memory consumers (if on Windows/Linux)
    try:
        print(f"\nTop Memory Consumers:")
        for proc in sorted(psutil.process_iter(['pid', 'name', 'memory_info']), 
                          key=lambda x: x.info['memory_info'].rss if x.info['memory_info'] else 0, 
                          reverse=True)[:5]:
            try:
                mem_mb = proc.info['memory_info'].rss / 1024 / 1024 if proc.info['memory_info'] else 0
                if mem_mb > 100:  # Only show processes using >100MB
                    print(f"  {proc.info['name']:<30} {mem_mb:>8.1f} MB")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception:
        pass

def clear_pandas_cache():
    """Clear pandas internal caches"""
    try:
        import pandas as pd
        # Clear any cached DataFrames
        pd.core.common._PANDAS_CACHE.clear()
        print("✓ Cleared pandas cache")
    except Exception as e:
        print(f"⚠ Could not clear pandas cache: {e}")

if __name__ == "__main__":
    print("=" * 50)
    print("MEMORY CLEANUP UTILITY")
    print("=" * 50)
    
    # Show initial stats
    show_memory_stats()
    
    # Clear memory
    print("\n" + "=" * 50)
    clear_memory()
    clear_pandas_cache()
    
    # Show final stats
    print("\n" + "=" * 50)
    show_memory_stats()
    
    print("\n✅ Memory cleanup complete!")
    print("\n💡 TIP: If memory is still high, try:")
    print("   1. Close other applications")
    print("   2. Restart Python/your IDE")
    print("   3. Reduce batch_size in interactive_analytics.py")
    print("   4. Process fewer tickers at once")


