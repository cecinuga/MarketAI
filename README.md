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



