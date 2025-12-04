import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, brier_score_loss

def stats_metrics(y_true_class, y_pred_prob, y_true_ret, y_pred_ret):
    y_pred_label = (np.array(y_pred_prob) > 0.5).astype(int)
    acc = accuracy_score(y_true_class, y_pred_label)
    f1 = f1_score(y_true_class, y_pred_label, zero_division=0)
    rmse = mean_squared_error(y_true_ret, y_pred_ret, squared=False)
    brier = brier_score_loss(y_true_class, y_pred_prob)
    return {'accuracy': acc, 'f1': f1, 'rmse': rmse, 'brier': brier}

def sharpe_ratio(returns, freq=252):
    r = np.array(returns)
    if len(r) < 2:
        return np.nan
    mu = np.nanmean(r)
    sd = np.nanstd(r)
    if sd == 0:
        return np.nan
    return (mu / sd) * np.sqrt(freq)

def max_drawdown(cum_returns):
    cr = np.array(cum_returns)
    peak = np.maximum.accumulate(cr)
    drawdown = (cr - peak) / peak
    return float(np.min(drawdown))
