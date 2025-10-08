# INT-L v1.0
Comprehensive Financial Data Platform with User Portfolio Management and Analyst Tools

## 🏗️ Database Architecture

### User-Centric Tables (Portfolio Management)
- **`users`** - User accounts and authentication
- **`user_deposits`** - Deposit transaction history
- **`bundles`** - Pre-defined investment bundles
- **`bundle_allocations`** - Hard-set allocations within bundles
- **`bundle_allocation_history`** - Bundle allocation versioning
- **`user_portfolio_allocations`** - User's chosen bundle allocations
- **`user_portfolio_history`** - Complete portfolio change audit trail
- **`bundle_performance`** - Calculated bundle NAV over time
- **`user_portfolio_performance`** - User's total portfolio performance

### Analyst-Centric Tables (Market Data & Analysis)
- **`stocks`** - Master stock registry (NYSE universe)
- **`stock_prices_daily`** - Daily OHLCV data for all stocks
- **`stock_prices_intraday`** - Intraday price data
- **`macro_indicators`** - Economic indicators from FRED
- **`macro_indicator_metadata`** - FRED series information
- **`sec_filings`** - SEC filing tracking (10-K, 10-Q, 8-K)
- **`corporate_events`** - Company event calendar
- **`stock_performance_metrics`** - Pre-calculated performance data
- **`sector_performance`** - Sector-level aggregation

## 🚀 Quick Start

### Prerequisites
- Aurora PostgreSQL database configured
- Python 3.7+ with required packages
- Environment variables set in `.env` file

### Database Setup
1. **Test Connection**: `python scripts/test_connection.py`
2. **Create All Tables**: `python scripts/create_all_tables.py`
3. **Verify Schema**: `python scripts/verify_schema.py`

### Scripts Overview
- **`test_connection.py`** - Database connectivity testing
- **`create_all_tables.py`** - Master script to create all tables
- **`verify_schema.py`** - Complete database schema verification
- **Individual table scripts** - 1-to-1 SQL/Python file mapping

## 📁 Project Structure
```
intl-v1/
├── scripts/                    # Python database scripts
│   ├── test_connection.py     # Connection testing
│   ├── create_all_tables.py   # Master table creation
│   ├── verify_schema.py       # Schema verification
│   └── [00-18]-pycreate-*.py  # Individual table scripts
├── sql/                       # SQL schema files
│   ├── 00-drop-stock_prices.sql
│   └── [01-18]-create-*.sql   # Individual table schemas
├── .env                       # Database credentials (not tracked)
└── .sqltools.json            # Database connection configuration
```

## 🔧 Features

### User Portfolio Management
- Bundle-based investing with hard-set allocations
- Complete portfolio audit trail
- Performance tracking across all timeframes
- User authentication and notification preferences

### Market Data & Analysis
- Complete NYSE stock universe tracking
- Daily and intraday price data
- Macro-economic indicators integration
- SEC filing and corporate event tracking
- Pre-calculated performance metrics
- Sector-level analysis

## 🎯 Next Steps
- Data ingestion pipeline development
- API integration for real-time data
- Portfolio performance calculation engine
- User interface development
