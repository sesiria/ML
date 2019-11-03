My XGBoost Implementation
it is a simple implementation contain two module
MyXGBoost.py
RegressionTree.py

1. The MyXGBoost contains the class MyXGBoostRegressor support to train multiple of regressionTreeReressor.
    support exact greedy approach for split
    the original second-order taylor expansion of the loss function
2. The RegressionTree contains the implementation of the class regressionTreeRegressor and the node class.
    the xgboost method to find the split point
    the standard deviation method to find the split point