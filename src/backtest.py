import numpy as np

def simple_backtest(signals, price_series, ret_series, commission=0.0002, slippage=0.0001):
    signals = np.array(signals)
    ret_series = np.array(ret_series)
    net_ret = np.zeros_like(ret_series)
    prev_signal = 0
    for i in range(len(signals)):
        sig = signals[i]
        gross = sig * ret_series[i]
        cost = 0.0
        if sig != prev_signal:
            cost = commission + slippage
        net_ret[i] = gross - cost
        prev_signal = sig
    cumulative = np.cumprod(1 + net_ret)
    return net_ret, cumulative
