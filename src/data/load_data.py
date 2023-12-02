from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import inspect
import torch
import os

class MovieRatingsDataset(torch.utils.data.Dataset):
    '''
    Dataset class for movie-lens dataset. Stores a matrix of movie ratings by users along \
    with the timestamps

    Attributes:
    users (ndarray): array of user ids representing elements in a sparse rating matrix
    items (ndarray): array of item ids representing elements in a sparse rating matrix
    ratings (ndarray): array of ratings 
    timestamp (ndarray): array of unix timestamps
    '''
    def __init__(self, df: pd.DataFrame):
        '''
        Create a MovieRatingsdataset object

        Parameters:
        df (DataFrame): the dataframe containing the data. Should have `user_id`, `item_id`, \
            `rating` and `timestamp` columns
        '''
        self.users = df['user_id'].to_numpy() - 1 # user_id's are indexed from 1
        self.items = df['item_id'].to_numpy() - 1 # item_id's are indexed from 1
        self.ratings = df['rating'].to_numpy().astype(np.float32)
        self.timestamps = df['timestamp'].to_numpy()

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, i: int) -> tuple[int, int, float]:
        return self.users[i], self.items[i], self.ratings[i]

def get_unique_num(key: str, path: str=None) -> int:
    '''
    Get number of unique entries in the dataset in the specified path for the specified column

    Parameters:
    key (str): column name where unique entries should be counted
    path (str): path to the tsv file containing the dataset. If not specified, dataset is read \
        by function `read_data_csv` using its default value for `path`
    
    Returns:
    The number of unique entries in the given dataset column
    '''
    if path is None: path = inspect.signature(read_data_csv).parameters['path'].default
    return len(read_data_csv(path)[key].unique())

def read_data_csv(path: str='data/raw/ml-100k/u.data') -> pd.DataFrame:
    '''Read the tsv file with user_id, item_id, rating and timestamp columns into a dataframe'''
    return pd.read_csv(path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

def get_crossval_datasets(
        path: str='data/raw/ml-100k/', 
        count: int=5, 
        batch_size: int=32, 
        train_format: str='u%d.base', 
        val_format: str='u%d.test') -> list[tuple[DataLoader, DataLoader]]:
    '''
    Get cross-validation training splits from the specified path for the movie-lens dataset

    Parameters:
    path (str): path to the dataset where the splits are located
    count (int): number of splits to load. The filenames are created using format parameters and 1-based index
    batch_size (int): batch_size for dataloaders
    train_format (str): format string for training data files. %d is replaced with the index of the split
    val_format (str): format string for validation data files. %d is replaced with the index of the split

    Returns:
    List of length `count` of tuples (train_dataloader, val_dataloader)
    '''
    return [
        (
            DataLoader(MovieRatingsDataset(read_data_csv(os.path.join(path, train_format % (i + 1)))), batch_size, shuffle=True),
            DataLoader(MovieRatingsDataset(read_data_csv(os.path.join(path, val_format % (i + 1)))), batch_size, shuffle=False)
        ) for i in range(count)
    ]
