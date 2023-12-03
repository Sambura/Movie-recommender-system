import sys; sys.path.append('.')

from src.data.load_data import get_crossval_datasets, create_dataloaders, get_unique_num
from src.models.simple_regressor import SimpleRegressorModelV1
from src.models.train import train_rsm
from src.utils.common import seed_everything
from src.models.evaluation import compute_metrics, make_markdown_table

from torch import nn
import torch
import os

dataset_splits = get_crossval_datasets()
dataloaders = create_dataloaders(dataset_splits)
output_path = 'models/simple_regressor_v1/'
os.makedirs(output_path, exist_ok=True)

randomizer_seed = 42
seed_everything(randomizer_seed)

metrics = []
loss_fn = nn.MSELoss()

for i, (train_dataloader, val_dataloader) in enumerate(dataloaders, 1):
    print(f'Training on split #{i}...')
    model = SimpleRegressorModelV1(get_unique_num('user_id'), get_unique_num('item_id'), 128, 128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train_loss, val_loss = train_rsm(model, optimizer, loss_fn, train_dataloader, val_dataloader)

    metrics.append(compute_metrics(model, val_dataloader))
    metrics[-1]['Training loss'] = train_loss[-1]

    model.save_model(os.path.join(output_path, f'model-128-split-{i}.pickle'))

print(f'Random seed: {randomizer_seed}')
print('Summary on cross validation training:')

print(make_markdown_table(metrics, return_markdown=False))
