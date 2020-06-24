# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:50:43 2019

@author: NeetKing
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')

dataset.head()

X = dataset.loc[:, ['Age', 'EstimatedSalary']].values
y = dataset['Purchased'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
logisticregression = LogisticRegression()
logisticregression.fit(X_train, y_train)

y_pred = logisticregression.predict(X_test)
print(y_pred)

y_compare = np.vstack((y_test,y_pred)).T

y_compare[:5,:]

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('true negatives (TN): Both, actual and predicted values are false: ', cm[0,0])
print('true positives (TP): Both, actual and predicted values are true: ', cm[1,1])
print('false positives (FP): Predicted value is yes but actual is false: ', cm[0,1])
print('false negative (FN): Predicted value is no but actual is true: ', cm[1,0])


from sklearn import metrics
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
print('The AUC value for the trained classifer on the test set is: ', metrics.auc(fpr, tpr))

from sklearn.metrics import roc_auc_score
print('Area under the ROC Curve: ' , roc_auc_score(y_test, y_pred))

from sklearn.metrics import average_precision_score
print('The Average Precision score for logistic clasifier is: ',average_precision_score(y_test, y_pred) )

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