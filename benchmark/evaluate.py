import sys; sys.path.append('.')

import pickle
from src.data.load_data import get_crossval_datasets, create_dataloaders
from src.models.evaluation import compute_metrics

model_path = 'models/simple_regressor_v1/model-128-split-1.pickle'
print(f'Evaluating model: {model_path}')

val_dataloader = create_dataloaders(get_crossval_datasets())[0][1]

with open(model_path, 'rb') as file:
    model = pickle.load(file)

print('Computing metrics...')
metrics = compute_metrics(model, val_dataloader)

print('Evaluation complete:')
for name, value in metrics.items():
    print(f'\t{name}: {value:0.3f}')
