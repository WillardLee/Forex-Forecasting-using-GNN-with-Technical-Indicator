import argparse
import yaml
import random
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch import nn
from src.data import download_eurusd, compute_indicators, build_graphs
from src.model import GATForecast
from src.metrics import stats_metrics, sharpe_ratio, max_drawdown
from src.backtest import simple_backtest

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def prepare_dataloaders(graphs, targets, train_idx, val_idx, test_idx, batch_size):
    def attach_targets(graph_list, tgt_list):
        new_list = []
        for g, t in zip(graph_list, tgt_list):
            d = g.clone()
            d.y_class = torch.tensor([t['dir']], dtype=torch.float32)
            d.y_ret = torch.tensor([t['ret_next']], dtype=torch.float32)
            d.y_vol = torch.tensor([t['vol_next']], dtype=torch.float32)
            new_list.append(d)
        return new_list
    train = attach_targets([graphs[i] for i in train_idx], [targets[i] for i in train_idx])
    val = attach_targets([graphs[i] for i in val_idx], [targets[i] for i in val_idx])
    test = attach_targets([graphs[i] for i in test_idx], [targets[i] for i in test_idx])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def train_epoch(model, loader, optim, device, loss_weights):
    model.train()
    total_loss = 0.0
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    for batch in loader:
        batch = batch.to(device)
        x = batch.x
        edge_index = batch.edge_index
        batch_vec = batch.batch if hasattr(batch, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=device)
        logits, pred_ret, pred_vol = model(x, edge_index, batch_vec)
        y_class = batch.y_class.view(-1).to(device)
        y_ret = batch.y_ret.view(-1).to(device)
        y_vol = batch.y_vol.view(-1).to(device)
        loss = loss_weights['class'] * bce(logits, y_class) + loss_weights['ret'] * mse(pred_ret, y_ret) + loss_weights['vol'] * mse(pred_vol, y_vol)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss += loss.item() * y_class.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    class_probs = []
    class_true = []
    ret_pred = []
    ret_true = []
    for batch in loader:
        batch = batch.to(device)
        x = batch.x
        edge_index = batch.edge_index
        batch_vec = batch.batch if hasattr(batch, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=device)
        logits, pred_ret, pred_vol = model(x, edge_index, batch_vec)
        probs = torch.sigmoid(logits).cpu().numpy()
        class_probs.extend(probs.tolist())
        class_true.extend(batch.y_class.view(-1).cpu().numpy().tolist())
        ret_pred.extend(pred_ret.cpu().numpy().tolist())
        ret_true.extend(batch.y_ret.view(-1).cpu().numpy().tolist())
    return {'class_prob': class_probs, 'class_true': class_true, 'ret_pred': ret_pred, 'ret_true': ret_true}

def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg['training'].get('seed', 123))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = download_eurusd(start=cfg['data']['start'])
    df_ind = compute_indicators(df)
    graphs, targets, dates = build_graphs(df_ind, window=cfg['data']['window'], k_neighbors=cfg['data'].get('k',2), horizon=cfg['data'].get('horizon',1))
    n = len(graphs)
    fold_size = cfg['cv']['fold_size']
    step = cfg['cv']['step']
    results = []
    for fold_start in range(0, n - fold_size, step):
        train_idx = list(range(0, fold_start))
        val_idx = list(range(fold_start, fold_start + fold_size))
        test_idx = [fold_start + fold_size]
        if not train_idx:
            continue
        train_loader, val_loader, test_loader = prepare_dataloaders(graphs, targets, train_idx, val_idx, test_idx, batch_size=cfg['training']['batch_size'])
        node_feat_len = graphs[0].x.size(1)
        model = GATForecast(node_feat_len=node_feat_len, gat_hidden=cfg['model']['gat_hidden'], gat_heads=cfg['model']['gat_heads']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['lr'])
        loss_weights = cfg.get('loss_weights', {'class':1.0,'ret':1.0,'vol':0.5})
        for epoch in range(cfg['training']['epochs']):
            train_loss = train_epoch(model, train_loader, optimizer, device, loss_weights)
        out = evaluate(model, test_loader, device)
        stats = stats_metrics(out['class_true'], out['class_prob'], out['ret_true'], out['ret_pred'])
        signal = (np.array(out['class_prob']) > 0.5).astype(int)
        net_ret, cumulative = simple_backtest(signal, None, out['ret_true'], commission=cfg['backtest']['commission'], slippage=cfg['backtest']['slippage'])
        sr = sharpe_ratio(net_ret)
        mdd = max_drawdown(cumulative)
        stats.update({'sharpe': sr, 'max_drawdown': mdd})
        results.append({'fold_start': fold_start, 'metrics': stats})
        print(f"Fold at {fold_start} metrics: {stats}")
    print("Walk-forward complete. Results:")
    for r in results:
        print(r)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
