from ..data.load_data import get_unique_num
from ..utils.common import get_device

import pandas as pd
import numpy as np
import torch

def recommend_items(model, user_id, train_dataset=None):
    if train_dataset is None: exclude_items = []
    else: exclude_items = train_dataset.df[train_dataset.df['user_id'] == user_id]['item_id'].values
    device = get_device(model)
    n_items = get_unique_num('item_id')
    items = torch.tensor([item_id for item_id in range(n_items) if item_id not in exclude_items], dtype=torch.int64, device=device)
    users = torch.tensor([user_id for _ in range(len(items))], dtype=torch.int64, device=device)
    predictions = model(users, items).view(-1)

    recommendations = pd.DataFrame(np.array([items.detach().cpu().numpy(), predictions.detach().cpu().numpy()]).T, columns=['item_id', 'score'])

    return recommendations.sort_values('score', ascending=False)
