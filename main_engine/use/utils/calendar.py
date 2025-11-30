"""
Trading Calendar Utilities

Handles trading dates, skipping weekends and holidays.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

# US trading calendar (NYSE/NASDAQ)
US_BUSINESS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())

def get_next_trading_dates(start_date, num_days):
    """
    Get the next N trading dates starting from start_date (exclusive).
    
    Args:
        start_date: Last trading date (datetime or string)
        num_days: Number of trading days to generate
    
    Returns:
        List of datetime objects (trading dates)
    """
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    
    # Start from the day after
    current = start_date + timedelta(days=1)
    
    # Generate trading dates
    trading_dates = pd.bdate_range(
        start=current,
        periods=num_days * 2,  # Generate extra to account for weekends/holidays
        freq=US_BUSINESS_DAY
    )[:num_days].tolist()
    
    return [pd.Timestamp(d) for d in trading_dates]

def is_trading_day(date):
    """Check if a date is a trading day"""
    if isinstance(date, str):
        date = pd.to_datetime(date)
    
    # Check if it's a weekday
    if date.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    # Check if it's a US federal holiday
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=date, end=date)
    
    return len(holidays) == 0

def get_last_trading_date(date):
    """Get the last trading date on or before the given date"""
    if isinstance(date, str):
        date = pd.to_datetime(date)
    
    # Go back up to 5 days to find a trading day
    for i in range(5):
        check_date = date - timedelta(days=i)
        if is_trading_day(check_date):
            return pd.Timestamp(check_date)
    
    # Fallback: just return the date
    return pd.Timestamp(date)

