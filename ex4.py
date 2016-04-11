from sklearn.neural_network import MLPClassifier
from openml.apiconnector import APIConnector
import numpy as np
from pandas import Series,DataFrame
import pandas as pd
import datetime



apikey = 'b6da739f426042fa9785167b29887d1a'
connector = APIConnector(apikey=apikey)
print 0
dataset = connector.download_dataset(554)

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



momentum_values = [0.3 , 0.7 ,0.9]
alpha_values = [0.1, 0.01 ,0.001]
hidden_layers= [20 , 50 ,100]

for hidden_layer in hidden_layers:
    for alpha in alpha_values:
        for mom in momentum_values:
            print " Hidden_layer neurons : " , hidden_layer ,  " ---alpha : " , alpha , " --- momentum : " , mom 
            mlp = MLPClassifier(hidden_layer_sizes=(hidden_layer,) , verbose=0, random_state=0, max_iter= 1000000, momentum=mom, algorithm= 'sgd', learning_rate='constant', alpha=alpha )
            start = datetime.datetime.now()
            mlp.fit(X_train, Y_train)
            print "Time elapsed: " , datetime.datetime.now() - start 
            print "Classification error" , 1 - mlp.score(X_test, Y_test) , "%"
            print "=============="


