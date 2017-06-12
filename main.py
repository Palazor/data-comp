# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler

# from pybrain.tools.shortcuts import buildNetwork
# from pybrain.supervised.trainers import BackpropTrainer
# from pybrain.structure.modules import SoftmaxLayer


def train_model(classifier, X):
    (x_trn, x_tst, y_trn, y_tst) = X
    classifier.fit(x_trn, y_trn)
    y_prd = classifier.predict(x_tst)

    acc = accuracy_score(y_tst, y_prd)
    pre = precision_score(y_tst, y_prd)
    rec = recall_score(y_tst, y_prd)
    f1 = f1_score(y_tst, y_prd)
    print(classifier, '\n\tacc:{}\n\tpre:{}\n\trec:{}\n\tf1:{}\n'.format(acc, pre, rec, f1))
    return classifier

train_data = pd.read_csv('./data/train_1.csv')
test_data = pd.read_csv('./data/test_1.csv')
desc = train_data.describe()
# print(desc)

train_count = train_data.shape[0]
features = train_data.iloc[:, :-1].append(test_data)
labels = train_data.iloc[:,-1]

imputer = Imputer(strategy="mean")
features_imp = imputer.fit_transform(features)

scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features_imp)

features_pd = pd.DataFrame(features_scaled, columns=features.columns)
cols = features_pd.shape[1]
for i in range(cols):
    features_pd['log_figure{}'.format(i + 1)] = np.log(features_pd['figure{}'.format(i + 1)] + 1)
# print(features_pd.describe())

features_train = features_pd.iloc[:train_count]
# print(features_train.describe())
features_test = features_pd.iloc[train_count:]
# print(features_test.describe())
x_train, x_test, y_train, y_test = train_test_split(features_train, labels, test_size=0.2, random_state=1)
X = (x_train, x_test, y_train, y_test)

svc = svm.SVC(kernel='rbf', C=10, random_state=1, gamma=0.1, max_iter=1000)
linear_svc = svm.LinearSVC(C=10, random_state=1, max_iter=100)
pred = train_model(svc, X).predict(features_test)
print(pred, len(pred), pred.mean())
train_model(linear_svc, X)

print(cross_val_score(svc, features_train, labels, scoring="neg_mean_squared_error", cv=10).mean())
print(cross_val_score(linear_svc, features_train, labels, scoring="neg_mean_squared_error", cv=10).mean())

# net = buildNetwork(2, 3, 1)