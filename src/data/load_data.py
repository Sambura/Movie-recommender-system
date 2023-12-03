from ..utils.common import replace_with_defaults

from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
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
    df (pd.DataFrame): an original dataframe
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
        self.df = df.copy()

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
    return len(read_data(**replace_with_defaults({'path': path}, read_data))[key].unique())

def read_data(path: str='data/raw/ml-100k/u.data') -> pd.DataFrame:
    '''Read the tsv file with user_id, item_id, rating and timestamp columns into a dataframe'''
    return pd.read_csv(path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

def read_user_data(path: str='data/raw/ml-100k/u.user') -> pd.DataFrame:
    '''Read the file with user_id, age, gender, occupation and zip_code columns into a dataframe'''
    return pd.read_csv(path, sep='|', names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])

def read_item_data(path: str='data/raw/ml-100k/u.item', drop_columns: bool=False) -> pd.DataFrame:
    '''Read the file with movie_id, title, release_date, video_release_date, url, and 19 genres columns into a dataframe'''
    item_data = pd.read_csv(path, sep='|', names=['movie_id', 'title', 'release_date', 'video_release_date', 'url', *[str(x) for x in range(19)]])
    if drop_columns: item_data.drop(columns=['video_release_date', 'url', 'title'], errors='ignore', inplace=True)
    return item_data

def get_genre_names(path: str='data/raw/ml-100k/u.genre') -> pd.DataFrame:
    '''Read the file with genre_name and genre_id columns into a dataframe'''
    return pd.read_csv(path, sep='|', names=['genre_name', 'genre_id'])

def get_crossval_paths(
        path: str='data/raw/ml-100k/', 
        count: int=5, 
        train_format: str='u%d.base', 
        val_format: str='u%d.test') -> list[tuple[str, str]]:
    '''
    Make a list of paths for loading data splits

    Parameters:
    path (str): path to the dataset where the splits are located
    count (int): number of splits
    train_format (str): format string for training data files. %d is replaced with the index of the split
    val_format (str): format string for validation data files. %d is replaced with the index of the split

    Returns:
    List of length `count` of tuples (train_path, val_path)
    '''
    return [(os.path.join(path, train_format % (i + 1)), os.path.join(path, val_format % (i + 1))) for i in range(count)]

def get_crossval_datasets(
        path: str=None, 
        count: int=None, 
        train_format: str=None, 
        val_format: str=None) -> list[tuple[MovieRatingsDataset, MovieRatingsDataset]]:
    '''
    Get cross-validation training splits from the specified path for the movie-lens dataset

    Parameters:
    Refer to `get_crossval_paths` for parameters description. If parameter value is not specified, \
        `get_crossval_paths` will use its default parameters.

    Returns:
    List of length `count` of tuples (train_dataset, val_dataset)
    '''
    paths = get_crossval_paths(**replace_with_defaults(locals(), get_crossval_paths))
    return [(get_dataset(train_path), get_dataset(val_path)) for train_path, val_path in paths]

def create_dataloaders(
        datasets: list[tuple],
        batch_size: int=32) -> list[tuple[DataLoader, DataLoader]]:
    '''
    Create a list of dataloaders from the list of datasets

    Parameters:
    datasets (list[tuple]): list of tuples of datasets (train_dataset, val_dataset). Dataloaders for train \
        datasets will be created with shuffle=True
    batch_size (int): batch_size for dataloaders

    Returns:
    List of length `count` of tuples (train_dataloader, val_dataloader)
    '''
    return [
        (
            DataLoader(train_dataset, batch_size, shuffle=True),
            DataLoader(val_dataset, batch_size, shuffle=False)
        ) for train_dataset, val_dataset in datasets
    ]

def get_dataset(path: str='data/raw/ml-100k/u.data') -> MovieRatingsDataset:
    return MovieRatingsDataset(read_data(path))
