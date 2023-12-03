import urllib.request
import zipfile
import os

def download_data(
        data_url: str="https://files.grouplens.org/datasets/movielens/ml-100k.zip", 
        data_dest: str="./data/raw/",
        zip_name: str="compressed.zip",
        verbose: bool=False) -> None:
    """
    Download dataset from the specified url and extract it in the given directory

    Parameters:
    data_url (str): Url to the zip archive to download
    data_dest (str): Path to directory where to put the downloaded file and where the files will be extracted.
        If the directory doesn't exist, it will be automatically created
    zip_name (str): How the downloaded archive should be named
    """
    zip_destination = os.path.join(data_dest, zip_name)

    os.makedirs(os.path.dirname(zip_destination), exist_ok=True)

    if verbose: print('Downloading...')
    if not os.path.exists(zip_destination):
        urllib.request.urlretrieve(data_url, zip_destination)

    if verbose: print('Extracting...')
    with zipfile.ZipFile(zip_destination, 'r') as zip_ref:
        zip_ref.extractall(data_dest)

    if verbose: print(f'Done. Data stored at {os.path.abspath(data_dest)}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("download_data")
    parser.add_argument('-u', '--data_url', default="https://files.grouplens.org/datasets/movielens/ml-100k.zip", type=str)
    parser.add_argument('-d', '--dest_path', default="./data/raw/", type=str)
    parser.add_argument('-z', '--zip_name', default="compressed.zip", type=str)
    args = parser.parse_args()
    download_data(data_url=args.data_url, data_dest=args.dest_path, zip_name=args.zip_name, verbose=True)
    