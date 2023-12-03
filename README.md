# Movie recommender system

This project focuses on creating an ML movie recommender system based on [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/).

## Author 
* Kirill Samburskiy (k.samburskiy@innopolis.university) : B20-RO-01

## Repo structure:

* `benchmark/`: contains scripts for evaluating trained models
* `notebooks/`: contains notebooks for data exploration, visualization, and model training
* `reports/`: reports regarding the work performed
* `src/`: python source code
    - `src/data/`: code for loading / processing data
    - `src/models/`: code for ML models and their training
    - `src/utils/`: util code used in other modules
* `references.md`: List of used references

## How to use

A short overview of how to use the code in this repo

### Download the dataset

Use python script at `src/data/download_data.py` to automatically download and extract the dataset to the default destination

### Train a model

You can run `src/models/run_training.py` script to launch the training process. Alternatively you can use `notebooks/2.0-training.ipynb` notebook. Training of simple models takes not more than 10 minutes for 5-fold cross-validation run on GPU.

### Pretrained models

You can also get pretrained model weights from [Github releases](https://github.com/Sambura/Movie-recommender-system/releases)

### Evaluation

Use `benchmark/evaluate.py` script to evaluate a trained model's performance.
