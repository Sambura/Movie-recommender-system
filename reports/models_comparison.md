# Performance of tested models

## Simple regressor V1

* embedding_sizes = 8:
    - Random seed: 42
    - Summary on cross validation training:
        ||Split 1|Split 2|Split 3|Split 4|Split 5|Average|
        |---|---|---|---|---|---|---|
        |MSA|0.7546|0.7474|0.7423|0.7407|0.7453|0.7461|
        |MSE|0.9119|0.8934|0.8834|0.8805|0.8793|0.8897|
        |RMSE|0.9549|0.9452|0.9399|0.9384|0.9377|0.9432|
        |F1|0.7459|0.7330|0.7251|0.7275|0.7311|0.7325|
        |training_loss|0.8547|0.8593|0.8600|0.8604|0.8622|0.8593|
    - Epoch count: 10
    - Optimizer: Adam (lr = 1e-3, wd = 1e-4)
    - MSELoss
    - Training time: ~ 6:50 min. (5 splits)

* embedding_sizes = 32:
    - Random seed: 42
    - Summary on cross validation training:
        ||Split 1|Split 2|Split 3|Split 4|Split 5|Average|
        |---|---|---|---|---|---|---|
        |MSA|0.7600|0.7498|0.7467|0.7453|0.7489|0.7502|
        |MSE|0.9236|0.9056|0.8922|0.8857|0.8866|0.8987|
        |RMSE|0.9610|0.9516|0.9446|0.9411|0.9416|0.9480|
        |F1|0.7422|0.7401|0.7216|0.7225|0.7266|0.7306|
        |Training loss|0.8737|0.8786|0.8801|0.8814|0.8811|0.8789|
    - Epoch count: 10
    - Optimizer: Adam (lr = 1e-3, wd = 1e-4)
    - MSELoss
    - Training time: ~ 8:35 min. (5 splits)

* embedding_sizes = 128:
    - Random seed: 42
    - Summary on cross validation training:
        ||Split 1|Split 2|Split 3|Split 4|Split 5|Average|
        |---|---|---|---|---|---|---|
        |MSA|0.7647|0.7502|0.7502|0.7454|0.7514|0.7524|
        |MSE|0.9323|0.9072|0.8983|0.8890|0.8897|0.9033|
        |RMSE|0.9656|0.9525|0.9478|0.9428|0.9432|0.9504|
        |F1|0.7399|0.7371|0.7152|0.7228|0.7244|0.7279|
        |Training loss|0.8847|0.8876|0.8888|0.8894|0.8907|0.8882|
    - Epoch count: 10
    - Optimizer: Adam (lr = 1e-3, wd = 1e-4)
    - MSELoss
    - Training time: ~ 8:35 min. (5 splits)

