# Logistic Regression

## The Iris flower dataset

### Link
[https://archive.ics.uci.edu/ml/datasets/iris](https://archive.ics.uci.edu/ml/datasets/iris)

### Features
1. sepal length in cm
2. sepal width in cm
3. petal length in cm
4. petal width in cm

### Category
- Iris Setosa
- Iris Versicolour
- Iris Virginica

### Preprocess
Since logistic regression is a binary classification, we set the class `Iris Setosa` as 1 and set both `Iris Versicolour` and `Iris Virginica` as 0.

### Training/Test set

|                 | Training | Test | Sum |
|-----------------|----------|------|-----|
| Iris Setosa     | 36       | 14   | 50  |
| Not Iris Setosa | 64       | 36   | 100 |
| Sum             | 100      | 50   | 150 |

## Performance
![Performance](https://raw.githubusercontent.com/LuChang-CS/plain-numpy-neural-networks/master/1.%20Logistic%20Regression/figs/performance.png)
