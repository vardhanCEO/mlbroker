<<<<<<< HEAD
# ALGOBOT - Algorithmic Trading System

A comprehensive algorithmic trading platform built with Flask, integrated with Zerodha Kite Connect API for live and paper trading.

## Features

- **Backtesting Engine**: Test strategies on historical data with comprehensive performance metrics
- **Live Trading**: Execute strategies in real-time with Zerodha integration
- **Paper Trading**: Risk-free testing mode for strategy validation
- **10+ Trading Strategies**: From basic to professional-grade with 55-70% win rates
- **Portfolio Management**: Diversified multi-strategy approach (9-16% monthly expected)
- **Risk Management**: Built-in stop-loss, take-profit, and position sizing
- **Web Dashboard**: Interactive UI for monitoring trades, P&L, and performance
- **Database Tracking**: SQLite database for trade history and analytics

## Available Strategies

### 🏆 Advanced High-Performance Strategies (RECOMMENDED)

| Strategy | Win Rate | Monthly Return | Best For | Timeframe |
|----------|----------|----------------|----------|-----------|
| **Multi-Timeframe Trend** | 65-70% | 3-5% | RELIANCE, TCS, INFY | 1-3 days |
| **VWAP + Supertrend** | 60-65% | 2-4% | Intraday trading | Same day |
| **Bollinger + RSI + ADX** | 70% | 4-7% | Swing trading | 2-7 days |
| **EMA + MACD + Stochastic** | 55-60% | 3-6% | Momentum | Various |
| **Conservative High Win** | 65-70% | 6-10% | Risk-averse | 15-min |

### 📊 Fundamental Strategies

- **High Growth Strategy**: Screen and trade high-profit-growth stocks
- **Portfolio Rotation**: Automated rotation based on fundamental metrics

### 🎯 Basic Strategies (For Learning)

- **MA Crossover**: Simple moving average crossover
- **RSI Strategy**: RSI-based mean reversion

**📖 For detailed strategy documentation:**
- [ADVANCED_STRATEGIES_GUIDE.md](ADVANCED_STRATEGIES_GUIDE.md) - Complete guide with examples
- [HIGH_GROWTH_STRATEGY_GUIDE.md](HIGH_GROWTH_STRATEGY_GUIDE.md) - Fundamental screening
- [STRATEGIES_QUICK_REFERENCE.md](STRATEGIES_QUICK_REFERENCE.md) - Quick reference guide

## Project Structure

```
algo-trading-app/
│
├── app.py                      # Main Flask application
├── config.py                   # Configuration management
├── requirements.txt            # Python dependencies
│
├── strategies/
│   ├── __init__.py
│   ├── base_strategy.py        # Base strategy class with indicators
│   └── ma_crossover.py         # MA Crossover & RSI strategies
│
├── backtesting/
│   ├── __init__.py
│   ├── backtest_engine.py      # Backtesting logic
│   └── data_handler.py         # Historical data fetching
│
├── trading/
│   ├── __init__.py
│   ├── zerodha_client.py       # Zerodha API integration
│   └── order_manager.py        # Order execution & management
│
├── database/
│   ├── __init__.py
│   ├── models.py               # Database models
│   └── db_manager.py           # Database operations
│
├── static/
│   ├── css/style.css           # Stylesheet
│   └── js/
│       ├── main.js             # Main JavaScript
│       ├── api.js              # API calls
│       └── chart.js            # Chart visualizations
│
└── templates/
    ├── base.html               # Base template
    ├── index.html              # Dashboard
    ├── backtest.html           # Backtesting interface
    ├── live_trading.html       # Live trading control
    └── strategies.html         # Strategy management
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Zerodha trading account (for live trading)
- Kite Connect API subscription (₹2000/month)

### Setup Steps

1. **Clone the repository**
```bash
git clone <repository-url>
cd ALGOBOT
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
cp .env.example .env
```

Edit `.env` file with your Zerodha credentials:
```env
KITE_API_KEY=your_api_key_here
KITE_API_SECRET=your_api_secret_here
KITE_USER_ID=your_zerodha_user_id

# Use paper mode for testing
TRADING_MODE=paper
```

5. **Initialize the database**
```bash
python -c "from app import app; from database.models import db; app.app_context().push(); db.create_all()"
```

6. **Run the application**
```bash
python app.py
```

7. **Access the dashboard**
Open your browser and navigate to: `http://localhost:5000`

## Zerodha Kite Connect Setup

1. **Create Kite Connect App**
   - Visit https://developers.kite.trade/
   - Create a new app
   - Note your API Key and API Secret

2. **Subscribe to Kite Connect**
   - Cost: ₹2000/month
   - Required for live trading

3. **Configure credentials**
   - Add API Key, Secret, and User ID to `.env` file

4. **Generate Access Token**
   - Use the login flow in the web dashboard
   - Or manually generate via `/api/auth/login` endpoint

## Usage

### Dashboard

The main dashboard provides:
- Today's and total P&L
- Win rate and performance metrics
- Open positions
- Recent trade history
- Real-time charts

### Backtesting

1. Navigate to **Backtest** page
2. Select a strategy (MA Crossover, RSI, etc.)
3. Choose symbol and date range
4. Set initial capital
5. Click "Run Backtest"
6. View results including:
   - Final capital and returns
   - Win rate and trade count
   - Sharpe ratio and risk metrics
   - Equity curve chart

### Live Trading

1. Navigate to **Live Trading** page
2. Select a strategy
3. Choose trading mode:
   - **Paper Trading**: Simulated trades (recommended for testing)
   - **Live Trading**: Real money (use with caution)
4. Click "Start Trading"
5. Monitor positions and P&L in real-time

### Strategy Management

View available strategies and their performance:
- Moving Average Crossover
- RSI Strategy
- Custom strategies

## API Endpoints

### Authentication
- `POST /api/auth/login` - Generate Kite login URL
- `POST /api/auth/callback` - Handle Kite callback

### Backtesting
- `POST /api/backtest/run` - Run backtest
- `GET /api/backtest/results/<id>` - Get backtest results
- `GET /api/backtest/history` - List all backtests

### Live Trading
- `POST /api/trade/start` - Start live trading
- `POST /api/trade/stop` - Stop live trading
- `GET /api/trade/status` - Get trading status
- `POST /api/trade/order` - Place manual order

### Dashboard
- `GET /api/dashboard/pnl` - Get P&L data
- `GET /api/dashboard/positions` - Current positions
- `GET /api/dashboard/trades` - Trade history
- `GET /api/dashboard/performance` - Performance metrics

### Strategies
- `GET /api/strategies` - List strategies
- `POST /api/strategies/create` - Create strategy
- `GET /api/strategies/<id>` - Get strategy details

## Configuration

### Risk Management Settings

Edit in `.env`:
```env
MAX_DAILY_LOSS=5000          # Maximum daily loss limit (₹)
MAX_POSITION_SIZE=50000      # Maximum position size (₹)
MAX_POSITIONS=5              # Maximum simultaneous positions
STOP_LOSS_PERCENTAGE=2       # Default stop loss (%)
TAKE_PROFIT_PERCENTAGE=5     # Default take profit (%)
```

### Trading Parameters

```env
DEFAULT_CAPITAL=100000       # Starting capital for backtesting
BACKTEST_COMMISSION=0.0003   # Commission rate (0.03%)
```

## Strategies

### Moving Average Crossover

Generates signals based on fast and slow MA crossovers.

**Parameters:**
- `fast_period`: Fast MA period (default: 20)
- `slow_period`: Slow MA period (default: 50)
- `ma_type`: SMA or EMA (default: SMA)

### RSI Strategy

Trades based on RSI overbought/oversold levels.

**Parameters:**
- `rsi_period`: RSI calculation period (default: 14)
- `oversold`: Oversold threshold (default: 30)
- `overbought`: Overbought threshold (default: 70)

## Development

### Adding New Strategies

1. Create new strategy file in `strategies/`
2. Inherit from `BaseStrategy`
3. Implement required methods:
   - `calculate_signals()`
   - `should_buy()`
   - `should_sell()`
4. Register in `app.py` STRATEGIES dict

Example:
```python
from strategies.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def __init__(self, parameters=None):
        super().__init__(name='MyStrategy', parameters=parameters)

    def calculate_signals(self, data):
        # Implement signal logic
        pass

    def should_buy(self, data, current_index):
        # Implement buy logic
        return False

    def should_sell(self, data, current_index):
        # Implement sell logic
        return False
```

### Running Tests

```bash
pytest tests/
```

## Security Best Practices

1. **Never commit API keys** - Use environment variables
2. **Use paper trading** - Test thoroughly before live trading
3. **Set risk limits** - Configure max loss and position limits
4. **Monitor actively** - Watch trades in real-time
5. **Regular backups** - Backup your database regularly

## Important Warnings

- **Trading involves risk** - You can lose money
- **Test thoroughly** - Always use paper trading mode first
- **No guarantees** - Past performance doesn't guarantee future results
- **Your responsibility** - You are responsible for all trading decisions

## Troubleshooting

### Database Issues
```bash
# Reset database
rm algobot.db
python -c "from app import app; from database.models import db; app.app_context().push(); db.create_all()"
```

### API Connection Issues
- Verify Kite API credentials
- Check access token validity
- Ensure network connectivity

### Module Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## License

This project is for educational purposes. Use at your own risk.

## Support

For issues and questions:
- Check existing issues on GitHub
- Review documentation
- Contact: [Your contact info]

## Disclaimer

This software is provided for educational purposes only. The creators assume no responsibility for financial losses incurred through the use of this software. Always trade responsibly and within your means.

---

**Built with Flask, Zerodha Kite Connect, and Chart.js**
"# mlbroker" 
"# mlbroker" 
=======
>>>>>>> 8c0d705 (first commit)
"# mlbroker" 
