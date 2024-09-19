# election-forecast

Forecasting the outcome of the upcoming Canadian federal election using python and scikit-learn.

## Overview

This application showcases a custom data pipeline that includes:

- Compiling historical riding-level election results for the previous 3 elections (2015, 2019, 2021) under the previous riding boundaries (2013-2023)

- Normalizing the riding-level results using the national totals from each election to produce characteristic scores for each riding (i.e. pro-BQ, pro-NDP, etc.)

- Combining the riding-level data with the most recent (2021) census data

- Training several different types of classification models to predict the winning party in each riding based on the normalized riding results and census data

- Using the best model, best data features, and an average of current polling data to predict the winner in each riding if the election was held now

- Projecting the forecast from the old (2013-2023) riding boundaries to the new (2024) riding boundaries

## Training

Using a typical 80-20 train-test split, each model performs well:

Ridge Classifier
```
 ridge           F1 (Train) = 0.9314     F1 (Test) = 0.9163

Confusion matrix - Test set
[[292  14   3   2   0   0]
 [  9 189   0   2   0   0]
 [  1   2  36   0   0   0]
 [  7   3   5  37   0   0]
 [  0   0   0   0   4   0]
 [  3   0   0   0   0   0]]
```

Logistic Regression
```
 logit           F1 (Train) = 0.9375     F1 (Test) = 0.9245

Confusion matrix - Test set
[[293  12   3   3   0   0]
 [  8 190   1   1   0   0]
 [  1   2  36   0   0   0]
 [  2   3   7  40   0   0]
 [  0   0   0   0   4   0]
 [  3   0   0   0   0   0]]
 ```

Random Forest
```
 rf              F1 (Train) = 0.9527     F1 (Test) = 0.9245

Confusion matrix - Test set
[[297  10   2   2   0   0]
 [ 10 188   1   1   0   0]
 [  0   2  37   0   0   0]
 [  5   3   7  37   0   0]
 [  0   0   0   0   4   0]
 [  3   0   0   0   0   0]]
```

Support Vector Classifier
```
 svc             F1 (Train) = 0.9170     F1 (Test) = 0.9031

Confusion matrix - Test set
[[290  15   5   1   0   0]
 [ 13 185   1   1   0   0]
 [  2   3  34   0   0   0]
 [  5   3   7  37   0   0]
 [  0   0   0   0   4   0]
 [  3   0   0   0   0   0]]
```

Simple Neural Network
```
 mlp             F1 (Train) = 0.9462     F1 (Test) = 0.9425

Confusion matrix - Test set
[[297  10   0   3   1   0]
 [  8 189   1   2   0   0]
 [  1   2  36   0   0   0]
 [  2   2   0  48   0   0]
 [  0   0   0   0   4   0]
 [  3   0   0   0   0   0]]
```

## Forecast

Despite the current poll lead enjoyed by CON, LIB is still projected to win the next election. This somewhat surprising result could be due to:
- the strong affinity many ridings have to a particular party
- the fact the training data (2015, 2019, 2021 elections) were favourable to LIB

### Example Output

Poll Average
```
{'CON': 42.69, 'LIB': 24.84, 'NDP': 18.42, 'BQ': 7.23, 'GRN': 4.08, 'OTH': 0.0}
```

Projection with `base_year = 2015`
```
winner     
BQ       53
CON     117
LIB     154
NDP      19
```

Projection with `base_year = 2019`
```
winner     
BQ       25
CON     130
LIB     178
NDP      10
```

Projection with `base_year = 2021`
```
winner     
BQ       46
CON     102
GRN       3
LIB     165
NDP      27
```
