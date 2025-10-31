#!/usr/bin/env python3
"""Check if breakout stocks are included in universe"""

from weekly_breakout_experiment import WeeklyBreakoutExperiment

def main():
    exp = WeeklyBreakoutExperiment()
    universe = exp._get_universe_list()
    
    print(f"📊 Total tickers in universe: {len(universe)}")
    print("\n🔍 Checking breakout stocks:")
    
    breakout_stocks = ['RGTI', 'IREN', 'QBTS', 'IONQ', 'APLD', 'VRT', 'TMC', 'IDR', 'MOD', 'NPCE']
    
    for stock in breakout_stocks:
        status = "✅" if stock in universe else "❌"
        print(f"  {stock}: {status}")
    
    print(f"\n📈 Breakout stocks included: {sum(1 for stock in breakout_stocks if stock in universe)}/10")

if __name__ == "__main__":
    main()
