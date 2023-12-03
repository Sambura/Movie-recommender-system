# Performance of tested models

## Simple regressor V1

* embedding_sizes = 32:
    - Train/val RMSE: $0.94 / 0.95$
    - Training losses: `[ 0.87438133 0.87776027 0.88042528 0.88068901 0.88349369 ]`
    - Validation losses: `[ 0.91940312 0.90061777 0.8911463  0.88885351 0.8873523 ]`
    - Epoch count: 10
    - Optimizer: Adam (lr = 1e-3, wd = 1e-4)
    - MSELoss
    - Training time: ~ 7:10 min. (5 splits)
    - No seed

* embedding_sizes = 8:
    - Train/val RMSE: $0.93 / 0.94$
    - Training losses: `[ 0.85636903 0.85841319 0.86104193 0.86042142 0.86222448 ]`
    - Validation losses: `[ 0.91189583 0.89338469 0.88427197 0.88024426 0.88076106 ]`
    - Epoch count: 10
    - Optimizer: Adam (lr = 1e-3, wd = 1e-4)
    - MSELoss
    - Training time: ~ 6:50 min. (5 splits)
    - No seed
