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

from pybrain.datasets import ClassificationDataSet
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.utilities import percentError


def train_model(classifier, X):
    (x_trn, x_tst, y_trn, y_tst) = X
    classifier.fit(x_trn, y_trn)
    y_prd = classifier.predict(x_tst)

    acc = accuracy_score(y_tst, y_prd)
    pre = precision_score(y_tst, y_prd)
    rec = recall_score(y_tst, y_prd)
    f1 = f1_score(y_tst, y_prd)
    # print(classifier, '\n\tacc:{}\n\tpre:{}\n\trec:{}\n\tf1:{}\n'.format(acc, pre, rec, f1))
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
x_train, x_test, y_train, y_test = train_test_split(features_train, labels, test_size=0.2, random_state=2)
X = (x_train, x_test, y_train, y_test)

svc = svm.SVC(kernel='rbf', C=10, random_state=1, gamma=0.1, max_iter=1000)
linear_svc = svm.LinearSVC(C=10, random_state=1, max_iter=100)
pred = train_model(svc, X).predict(features_test)
# print(pred, len(pred), pred.mean())
train_model(linear_svc, X)

# print(cross_val_score(svc, features_train, labels, scoring="neg_mean_squared_error", cv=10).mean())
# print(cross_val_score(linear_svc, features_train, labels, scoring="neg_mean_squared_error", cv=10).mean())



dsTrain = ClassificationDataSet(18, 1, nb_classes=2)
rows = len(x_train)
for row in range(rows):
    dsTrain.addSample(tuple(x_train.iloc[row]), y_train.iloc[row])
dsTrain._convertToOneOfMany()

dsTest = ClassificationDataSet(18, 1, nb_classes=2)
rows = len(x_test)
for row in range(rows):
    dsTest.addSample(tuple(x_test.iloc[row]), y_test.iloc[row])
dsTest._convertToOneOfMany()

if True:
    fnn = buildNetwork(18, 15, 2, outclass=SoftmaxLayer)
    trainer = BackpropTrainer(fnn, dataset=dsTrain, momentum=0, verbose=False, weightdecay=0.0)

    # trainer.trainEpochs(1000)
    trainer.trainUntilConvergence(maxEpochs=100000)
    NetworkWriter.writeToFile(fnn, 'filename.xml')
else:
    fnn = NetworkReader.readFrom('filename.xml')
    trainer = BackpropTrainer(fnn, dataset=dsTrain, momentum=0, verbose=False, weightdecay=0.0)

trnresult = percentError(trainer.testOnClassData(), dsTrain['target'])
testResult = percentError(trainer.testOnClassData(dataset=dsTest), dsTest['target'])
# print("epoch: {}\ntrain error: {}\ntest error: {}".format(trainer.totalepochs, trnresult, testResult))

res = trainer.testOnClassData(dataset=dsTest)
out = fnn.activateOnDataset(dsTest).argmax(axis=1)
tgt = [int(x[1]) for x in dsTest['target']]
# print(accuracy_score(tgt, res),
#       precision_score(tgt, res),
#       recall_score(tgt, res),
#       f1_score(tgt, res))

pred = train_model(svc, X).predict(x_test)
print('Test', f1_score(tgt, pred), f1_score(tgt, res), f1_score(tgt, out))

res = trainer.testOnClassData(dataset=dsTrain)
out = fnn.activateOnDataset(dsTrain).argmax(axis=1)
tgt = [int(x[1]) for x in dsTrain['target']]
# print(accuracy_score(tgt, res),
#       precision_score(tgt, res),
#       recall_score(tgt, res),
#       f1_score(tgt, res))

pred = train_model(svc, X).predict(x_train)
print('Train', f1_score(tgt, pred), f1_score(tgt, res), f1_score(tgt, out))
