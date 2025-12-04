# Graph-based EUR/USD Swing Forecasting (GAT)

This repository contains a reproducible proof-of-concept for a graph-based forecasting framework for the EUR/USD currency pair in a swing-trading horizon (daily/weekly).

Key ideas:
- Technical indicators are nodes (RSI(14), MACD(12,26,9), EMAs).
- Dynamic graphs: rolling windows construct edges (rolling correlation or k-NN).
- A Graph Attention Network (GAT) learns temporal dynamics and cross-indicator relationships.
- Model outputs: (i) direction probability (up/down), (ii) next-period return, (iii) an estimated volatility range.
- Strict walk-forward validation with leakage controls.
- Benchmarks: Random Walk, 12/26 EMA crossover, LSTM baseline.
- Evaluation: Accuracy, F1, RMSE, Brier, Sharpe ratio, max drawdown. Backtests include transaction costs and slippage.

This is an academic proof-of-concept — not a trading recommendation. No live trading code.

Quickstart:
1. Create a Python environment and install requirements: pip install -r requirements.txt
2. Run training (example): python -m src.train --config configs/default.yaml
3. Use notebooks/ for exploration and reproducibility.

Ethics note: This project is for research and educational purposes only. Do not use the code to execute live trades without rigorous additional validation and legal/compliance checks.