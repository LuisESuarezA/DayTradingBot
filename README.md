# Discrete Trading Environment with Alpaca API

This project implements a **Reinforcement Learning-based Trading Environment** using the Alpaca API. It allows agents to learn and test trading strategies for a specified stock ticker over historical data.

## Features
- Discrete action space for trading (`Buy`, `Hold`, `Sell`).
- Observation space with historical OHLCV (Open, High, Low, Close, Volume) data, cash balance, and shares held.
- Integration with the Alpaca API for historical stock data.
- Model training using the `Stable-Baselines3` library.
- Visualization of trading performance (stock price, shares held, portfolio value).

## Prerequisites

### Libraries and Dependencies
- `os`
- `gym`
- `numpy`
- `pandas`
- `matplotlib`
- `alpaca-trade-api`
- `stable-baselines3`

Install dependencies using:
```bash
pip install gym numpy pandas matplotlib alpaca-trade-api stable-baselines3
```

### Alpaca API Keys
Create an account on [Alpaca](https://alpaca.markets/) and obtain the following:
- `ALPACA_API_KEY_ID`
- `ALPACA_API_SECRET_KEY`

Replace the placeholder keys in the code:
```python
ALPACA_API_KEY_ID = 'Your_API_Key'
ALPACA_API_SECRET_KEY = "Your_Secret_Alpaca_Key"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
```

## Code Structure

### 1. DiscreteTradingEnv Class
Custom trading environment implementing the OpenAI Gym API:
- **Initialization Parameters:**
  - `ticker`: Stock ticker symbol (e.g., "AAPL").
  - `initial_cash`: Starting cash amount.
  - `lookback_window`: Number of historical steps to include in observations.
  - `start_date`, `end_date`: Date range for historical data.
  - `timeframe`: Data granularity (e.g., '1Hour').

- **Core Methods:**
  - `reset()`: Initializes the environment.
  - `step(action)`: Executes the given action (0 = Sell, 1 = Hold, 2 = Buy).
  - `render()`: Prints the current state of the portfolio.
  - `_fetch_historical_data()`: Fetches historical OHLCV data from the Alpaca API.

### 2. Training and Testing
- **Training:**
  A `DQN` (Deep Q-Network) model is trained using `Stable-Baselines3`.

- **Testing:**
  The trained model is tested on a separate dataset to evaluate performance. Metrics such as stock price, shares held, and portfolio value are plotted over time.

## Running the Code

1. **Set up the environment:**
   Replace the Alpaca API keys in the code.

2. **Train the model:**
   The `DQN` model is trained on the historical data specified in `make_train_env()`.

3. **Test the model:**
   The trained model is tested on out-of-sample data, and performance graphs are generated.

Run the script:
```bash
python trading_env.py
```

## Output

### Graphs
- **Stock Price:** Tracks the stock price over time.
- **Shares Held:** Tracks the number of shares held over time.
- **Portfolio Value:** Tracks the portfolio value over time.

### Console Logs
Detailed logs of each step:
```
Step: 10, Action: 2, Shares: 50, Cash: 5000.00, Portfolio: 10500.00, Reward: 500.00
```

### Final Portfolio Value
Displays the final portfolio value at the end of the test period:
```
Portfolio Value: 120000.00
```

## Notes
- The code includes a bonus reward for achieving a portfolio value of 120% of the initial cash.
- Modify parameters (e.g., `lookback_window`, `timeframe`) to customize the environment.

## License
This project is licensed under the MIT License.

---
Happy trading! 🚀
