#from sklearn.neural_network import MLPClassifier
from openml.apiconnector import APIConnector
import numpy as np
from pandas import Series,DataFrame
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from pandas.tools.plotting import parallel_coordinates
from sklearn import preprocessing
plt.style.use('ggplot')




from pandas import read_csv

from pandas.tools.plotting import parallel_coordinates

data = pd.read_csv('results.csv')
data['aa'] = 1 
parallel_coordinates(data ,'aa')
plt.show()

apikey = 'b6da739f426042fa9785167b29887d1a'
connector = APIConnector(apikey=apikey)

dataset = connector.download_dataset(554)


columns_names = [  'feature_' + str(x) for x in range(0,784) ]
columns_names.append('target')
train = dataset.get_dataset()
train = pd.DataFrame(train , columns = columns_names ) 
y = train['target']
X = train.iloc[:,:-1]

X_train = X.iloc[0:60000].values
Y_train = y.iloc[0:60000].values
X_test = X.iloc[60000:].values
Y_test = y.iloc[60000:].values


mlp = MLPClassifier(hidden_layer_sizes=(50,) , verbose=0, random_state=0, max_iter= 1000000, momentum=0.9, algorithm= 'sgd', learning_rate='constant', alpha=0.001 )
start = datetime.datetime.now()
mlp.fit(X_train, Y_train)
print(datetime.datetime.now() - start )
print(mlp.score(X_test, Y_test))


