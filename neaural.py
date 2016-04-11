from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.core import Dense, Activation
from openml.apiconnector import APIConnector
import numpy as np
from pandas import Series,DataFrame
import pandas as pd

apikey = 'b6da739f426042fa9785167b29887d1a'
connector = APIConnector(apikey=apikey)
print 0
dataset = connector.download_dataset(554)
optimizer=None
exception_verbosity='high'

print 1 
columns_names = [  'feature_' + str(x) for x in range(0,784) ]
columns_names.append('target')
print 2
train = dataset.get_dataset()
train = pd.DataFrame(train , columns = columns_names ) 
y = train['target']
X = train.iloc[:,:-1]

X_train = X.iloc[0:60000].values
Y_train = y.iloc[0:60000].values
X_test = X.iloc[60000:].values
Y_test = y.iloc[60000:].values

from keras.utils import np_utils, generic_utils

y_train, y_test = [np_utils.to_categorical(x) for x in (Y_train, Y_test     )]

print X_train.shape
print Y_train.shape
print X_test.shape
print Y_test.shape

print X_train
model = Sequential()


model.add(Dense(   784 , 50 , init="glorot_uniform"))
model.add(Activation("relu"))
model.add(Dense(50,10 , init="glorot_uniform"))
model.add(Activation("softmax"))


#model.compile(loss='categorical_crossentropy', optimizer='sgd')
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001, momentum=0.9, nesterov=True))
model.fit(X_train, Y_train, nb_epoch=5, batch_size=32)
objective_score = model.evaluate(X_test, Y_test, batch_size=32)

print objective_score