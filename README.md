# ML - Logistic Regression

## Overview

Logistic regression is a widely used method because it allows the modeling of binary variables or sums of binary variables. It is widely used in the medical field (healing or not of a patient), in sociology, in epidemiology, in quantitative marketing (purchase or not of products or services following an action) and in finance for risk modeling (scoring).

Principles of logistic regression

The principle of the logistic regression model is to relate the occurrence or non-occurrence of an event to the level of explanatory variables. For example, in the phytosanitary field, it is sought to evaluate from which dose of a chemical agent, an insect will be neutralized.

###  Importing the libraries 

* numpy `import numpy as np` 
* matplotlib `import matplotlib.pyplot as plt`
* pandas `import pandas as pd`

### Importing the dataset

```python
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values
```
### Split data

first we split the dataset into training and test data

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
```
To work with the logistic regression, we must go through the operation of standardization of each independent variable

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```
### Form a classifier: Adjust the regression model to the training data and predict the test data.
Now that we have our features and our learning data, we can train a classifier to try to predict the category of a new critic.

```python
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
```
Predicting the Test set results

```python
y_pred = classifier.predict(X_test)
```

### Making the Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
```
### View the result

```python
# Visualising the Training set results 
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X, Y = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

probs = logisticregression.predict(np.array([X.ravel(), Y.ravel()]).T).reshape(X.shape)
plt.contourf(X, Y, probs, alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X.min(), X.max())
plt.ylim(Y.min(), Y.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j, edgecolor="white")

plt.title('Logistic Regression Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
```
![](/imgs/view-result.png?raw=true)
```python
# Visualising the Test set results 
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X, Y = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

probs = logisticregression.predict(np.array([X.ravel(), Y.ravel()]).T).reshape(X.shape)
plt.contourf(X, Y, probs, alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X.min(), X.max())
plt.ylim(Y.min(), Y.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j, edgecolor="white")

plt.title('Logistic Regression Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
```
![](/imgs/view-result2.png?raw=true)

Enjoy the codes.
