# OrderAi
A OpenSource software written in python using tensorflow, dx_feed and another API for algorithmic trading using Linear Regression applied to Neaural Networks
Taking advantage of the orderflows we will predict the market price

## Neural Network
The structure of the network is the following, 5 layers with `X_train.shape[0]` neurons.
`Relu X Relu X Relu X Sigmoid X Identity`


### Loss Function
The Loss function is computed using `MeanSquarredError` + `Lasso`


### Gradient Descend
The gradient is computed using `Adam`


### Evaluation
The evalutation is computed using the following metrics:
1) `MeanSquarredError`
2) `MeanAbsoluteError`
3) `Residual`
4) `R2`


### List of Optimizations Features
1) `MiniBatch`
2) `Normalization`
3) `Lasso`
4) `Cross-Validation`
5) `Any simple optimization...`


### Stats
#### The following statistics are available with the following options:
Epochs: `5000`


N Samples: `3000`, N Features: `2`
Learning Rate: `0.01`
Batch Size: `250`
Noise: `10`
Cross-Validation iterations: `1`

Original:  `86.98843005885753`
Predicted:  `73.84910318525165` 

Train_MeanSquaredError:  `97.680115`
Train_MeanAbsoluteError:  `7.9298763`
Train_MeanSquaredLogError:  `0.0`
Train_R2Accuracy:  `0.9879063843219044`
Test_MeanSquaredError:  `103.46464`
Test_MeanAbsoluteError:  `8.246868`
Test_MeanSquaredLogError:  `0.0`
Test_R2Accuracy:  `0.9867968104336003`