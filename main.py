# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.externals import joblib
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
x_train, x_test, y_train, y_test = train_test_split(features_train, labels, test_size=0.2)#, random_state=1)
X = (x_train, x_test, y_train, y_test)

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

svc = None
fnn = None
if False:
    svc = svm.SVC(kernel='rbf', C=10, random_state=1, gamma=0.1, max_iter=1000)
    linear_svc = svm.LinearSVC(C=10, random_state=1, max_iter=100)
    pred = train_model(svc, X).predict(features_test)
    # print(pred, len(pred), pred.mean())
    train_model(linear_svc, X)
    joblib.dump(svc, 'svc.pkl')

    fnn = buildNetwork(18, 15, 2, outclass=SoftmaxLayer)
    trainer = BackpropTrainer(fnn, dataset=dsTrain, momentum=0, verbose=False, weightdecay=0.0)
    trainer.trainUntilConvergence(maxEpochs=100000)
    NetworkWriter.writeToFile(fnn, 'ann.xml')
else:
    svc = joblib.load('svc.pkl')

    fnn = NetworkReader.readFrom('ann.xml')
    trainer = BackpropTrainer(fnn, dataset=dsTrain, momentum=0, verbose=False, weightdecay=0.0)

trnresult = percentError(trainer.testOnClassData(), dsTrain['target'])
testResult = percentError(trainer.testOnClassData(dataset=dsTest), dsTest['target'])
# print("epoch: {}\ntrain error: {}\ntest error: {}".format(trainer.totalepochs, trnresult, testResult))

tgt_tst = [int(x[1]) for x in dsTest['target']]
out_tst = fnn.activateOnDataset(dsTest).argmax(axis=1)
prd_tst = svc.predict(x_test)
svc_f1_tst = f1_score(tgt_tst, prd_tst)
fnn_f1_tst = f1_score(tgt_tst, out_tst)

tgt_trn = [int(x[1]) for x in dsTrain['target']]
out_trn = fnn.activateOnDataset(dsTrain).argmax(axis=1)
prd_trn = svc.predict(x_train)
svc_f1_trn = f1_score(tgt_trn, prd_trn)
fnn_f1_trn = f1_score(tgt_trn, out_trn)
print("\t\t  SVC\t\t  ANN\t\tWIN")
print('Test\t{0:.5}\t\t{1:.5}\t\t{2}'.format(svc_f1_tst, fnn_f1_tst, 'SVC' if svc_f1_tst > fnn_f1_tst else 'FNN'))
print('Train\t{0:.5}\t\t{1:.5}\t\t{2}'.format(svc_f1_trn, fnn_f1_trn, 'SVC' if svc_f1_trn > fnn_f1_trn else 'FNN'))
