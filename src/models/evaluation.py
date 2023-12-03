from ..utils.common import get_device

from torch.utils.data import DataLoader
from IPython.display import Markdown
import sklearn.metrics as metrics
from torch import nn
import numpy as np
import torch

def compute_metrics(model: nn.Module,
                    val_dataloader: DataLoader) -> dict:
    device = get_device(model)
    model.eval()

    predictions, true_ratings = [], []
    with torch.no_grad():
        for users, items, ratings in val_dataloader:
            users, items, ratings = users.to(device), items.to(device), ratings.to(device)
            output = model(users, items).view(-1)
            
            predictions += output.detach().cpu().numpy().tolist()
            true_ratings += ratings.detach().cpu().numpy().tolist()

    predictions = np.array(predictions)
    true_ratings = np.array(true_ratings)

    msa = metrics.mean_absolute_error(true_ratings, predictions)
    mse = metrics.mean_squared_error(true_ratings, predictions)
    rmse = np.sqrt(mse)
    f1 = metrics.f1_score(true_ratings >= 3.5, predictions >= 3.5)
    
    return {
        'MSA': msa, 'MSE': mse, 'RMSE': rmse, 'F1': f1
    }

def make_markdown_table(metrics, return_markdown=True):
    metrics_table = '||'

    for i, metric in enumerate(metrics, 1):
        metrics_table += f'Split {i}|'
    metrics_table += 'Average|\n|---|'

    for i, metric in enumerate(metrics, 1):
        metrics_table += '---|'
    metrics_table += '---|'

    for metric_name in metrics[0]:
        metric_list = []
        metrics_table += f'\n|{metric_name}|'
        for i, metric in enumerate(metrics, 1):
            metric_list.append(metric[metric_name])
            metrics_table += f'{metric[metric_name]:0.4f}|'
        metrics_table += f'{np.mean(metric_list):0.4f}|'

    return Markdown(metrics_table) if return_markdown else metrics_table
