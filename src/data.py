"""Data loader, indicator calculator, and dynamic graph builder.

Basic flow:
- load OHLCV (default: yfinance 'EURUSD=X')
- compute indicators (RSI(14), MACD, EMAs)
- for each time index t, create a graph representing the last W observations:
    - nodes: indicator series (optionally include price)
    - node features: recent window of each indicator (or stats)
    - edges: computed via rolling correlation or k-NN on indicator windows
- returns sequences of (graph, target) for supervised training
"""
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from sklearn.neighbors import NearestNeighbors
import torch
from torch_geometric.data import Data
from typing import List, Tuple

def download_eurusd(start='2010-01-01', end=None, ticker='EURUSD=X'):
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df.dropna()
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['rsi14'] = RSIIndicator(out['Close'], window=14).rsi()
    macd = MACD(out['Close'], window_slow=26, window_fast=12, window_sign=9)
    out['macd'] = macd.macd()
    out['macd_signal'] = macd.macd_signal()
    out['ema12'] = EMAIndicator(out['Close'], window=12).ema_indicator()
    out['ema26'] = EMAIndicator(out['Close'], window=26).ema_indicator()
    out['ret_1'] = out['Close'].pct_change()
    out['vol_20'] = out['ret_1'].rolling(20).std()
    out = out.dropna()
    return out

def _series_to_node_feat(series: np.ndarray):
    # Input: (window,), output: (window,) normalized
    s = series.astype(np.float32)
    mu = np.nanmean(s)
    sd = np.nanstd(s) + 1e-9
    return (s - mu) / sd

def build_graphs(df_indicators: pd.DataFrame, window: int = 30, k_neighbors: int = 2, horizon: int = 1) -> Tuple[List[Data], List[dict], List[pd.Timestamp]]:
    """
    Build list of torch_geometric.data.Data graphs and targets.

    Each graph uses the past `window` observations to create node features for each indicator.
    Node features are the normalized time-series (shape: window). Edges are constructed using k-NN on abs(correlation)
    computed across the window.

    Returns:
    - list of Data objects (with x: [num_nodes, window], edge_index: [2, E])
    - list of target dicts: {'ret_next': float, 'vol_next': float, 'dir': int}
    - list of dates aligned to the graph (the last date used to build the graph)
    """
    indicator_cols = ['rsi14', 'macd', 'macd_signal', 'ema12', 'ema26']
    graphs = []
    targets = []
    dates = []
    vals = df_indicators[indicator_cols].values
    ret_series = df_indicators['ret_1'].values
    dates_index = df_indicators.index

    num_samples = len(df_indicators) - window - horizon + 1
    for i in range(num_samples):
        start = i
        end = i + window  # exclusive end index for window slice
        window_df = df_indicators.iloc[start:end]
        node_feats = []
        for col in indicator_cols:
            series = window_df[col].values
            feat = _series_to_node_feat(series)
            node_feats.append(feat)
        node_feats = np.stack(node_feats, axis=0)  # (num_nodes, window)
        # Build edges via corr
        corr = np.corrcoef(node_feats)
        # Ensure finite
        corr = np.nan_to_num(corr)
        dist = 1 - np.abs(corr)
        # k-NN
        k = min(k_neighbors + 1, dist.shape[0])
        nbrs = NearestNeighbors(n_neighbors=k, metric='precomputed').fit(dist)
        _, idxs = nbrs.kneighbors(dist)
        edges = []
        for src in range(dist.shape[0]):
            for dst in idxs[src]:
                if src != dst:
                    edges.append([src, dst])
        if len(edges) == 0:
            edge_index = np.zeros((2, 0), dtype=np.int64)
        else:
            edge_index = np.array(edges).T.astype(np.int64)
        # Targets (aligned to end index)
        target_idx = end + horizon - 1
        ret_next = float(ret_series[target_idx])
        vol_next = float(df_indicators['ret_1'].iloc[end: end + horizon].std())
        # Direction binary (1 if up, 0 if down or zero)
        dir_label = int(ret_next > 0)
        # Create torch_geometric Data
        x = torch.tensor(node_feats, dtype=torch.float)  # (num_nodes, window)
        edge_index_t = torch.tensor(edge_index, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index_t)
        graphs.append(data)
        targets.append({'ret_next': ret_next, 'vol_next': vol_next, 'dir': dir_label})
        dates.append(dates_index[end - 1])  # last date used for the graph
    return graphs, targets, dates
"""