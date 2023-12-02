from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
import numpy as np
import torch

def train_rsm(model: nn.Module, 
              optimizer: torch.optim.Optimizer, 
              loss_fn: nn.Module, 
              train_dataloader: DataLoader, 
              val_dataloader: DataLoader, 
              device: torch.device=None, 
              epoch_count: int=10):
    '''
    Train a recommender system model.

    Parameters:
    model (nn.Module): The model to train. It will be automatically sent to the training device
    optimizer (torch.optim.Optimizer): The optimizer to use 
    loss_fn (nn.Module): Loss function to use
    train_dataloader (data.DataLoader): Dataloader to use for training
    val_dataloader (data.Dataloader): Dataloader to use for validation
    device (torch.device): Device where the training should be performed. If None, selected automatically
    epoch_count (int): Number of epochs to run for
    '''
    if device is None: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    def train_loop(dataloader, train):
        if train: model.train()
        else: model.eval()
        total_loss, total = 0, 0
        # how to add no_grad??
        with tqdm(dataloader, total=len(dataloader), desc=f'{"Training" if train else "Validation"} epoch {epoch_idx + 1}', leave=False) as pbar:
            for users, items, ratings in pbar:
                if train: optimizer.zero_grad()
                users, items, ratings = users.to(device), items.to(device), ratings.to(device)

                output = model(users, items)

                loss = loss_fn(output.view(-1), ratings)
                total_loss += loss.item() * len(ratings)
                total += len(ratings)

                pbar.set_postfix({'loss': total_loss / total, 'rmse': np.sqrt(total_loss / total)})
                
                if train:
                    loss.backward()
                    optimizer.step()
        
        return total_loss / total
    
    train_losses, val_losses = [], []
    for epoch_idx in range(epoch_count):
        train_losses.append(train_loop(train_dataloader, train=True))
        val_losses.append(train_loop(val_dataloader, train=False))

    return train_losses, val_losses
