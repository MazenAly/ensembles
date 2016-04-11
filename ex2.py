from openml.apiconnector import APIConnector
import numpy as np
from pandas import Series,DataFrame
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
import datetime
plt.style.use('ggplot')
from sklearn.utils import resample
from sklearn.cross_validation import cross_val_score
from collections import Counter
from sklearn import tree
apikey = 'b6da739f426042fa9785167b29887d1a'
connector = APIConnector(apikey=apikey)
dataset = connector.download_dataset(44)

columns_names = [  'feature_' + str(x) for x in range(0,57) ]
columns_names.append('target')
train = dataset.get_dataset()
train = pd.DataFrame(train , columns = columns_names ) 
y = DataFrame(train['target'],  columns = ['target'])
X = train.iloc[:,:-1]
bootstrap_nums =100

def loss_func(x):
    loss_counts = [ i for i in x if i != x['target'] and  i == i  ]
    predictions_no = len([ i for i in x if i == i  ]) - 1
    if predictions_no == 0:
        return -1
    return predictions_no/float(bootstrap_nums) * (len(loss_counts) / float(predictions_no))

def variance_func(x):
    x = x.drop('bias')
    x = x[1:]
    variantes = [ i for i in x if  i == i  ]
    predictions_no = len(variantes)
    if predictions_no == 0:
        return -1
    variance = predictions_no/float(bootstrap_nums) *  (1 - (Counter(variantes).most_common(1)[0][1] / float(predictions_no)))
    return variance

def diff(a ,b):
    b= set(b)
    return [ x for x in a if x not in b ]   


#Bias, Variance and AUC for a full CART decision tree

var_bias_df = DataFrame(y , columns = ['target' ] ) 
for i in range(bootstrap_nums):
    training_samples,  training_labels = resample(X, y)
    test_index = diff(X.index , training_samples.index.unique() )
    test_samples = X.iloc[test_index,:]
    test_labels =  y.iloc[test_index] 
    clf = tree.DecisionTreeClassifier()
    clf.fit(training_samples, training_labels)
    preds = clf.predict(test_samples)
    booster_preds = DataFrame( preds , index=test_samples.index , columns = ['booster_' + str(i) ]) 
    var_bias_df = var_bias_df.join(booster_preds)
    

var_bias_df['bias'] = var_bias_df.apply(  loss_func ,  axis = 1)
var_bias_df['var'] = var_bias_df.apply(  variance_func ,  axis = 1)
bias_list = var_bias_df['bias'].values.tolist()
var_list = var_bias_df['var'].values.tolist()
print "CART bias: " , np.mean(list(filter(lambda x: x!= -1  , bias_list)))
print "CART Variance: " , np.mean(list(filter(lambda x: x!= -1, var_list)))



clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y.target.values)
auc = cross_val_score(clf, X, y.target.values, cv=10 , scoring='roc_auc')
print "CART AUC: " ,auc.mean()


#OOB for RandomForest of different number of trees

start = datetime.datetime.now()
estimators_nums = [ 1,2,4,8,16,32,64,128]
out_of_bag_errors = []
for j , est_num in enumerate(estimators_nums):
    print est_num
    clf = RandomForestClassifier(n_estimators=est_num , max_features='auto' , bootstrap=True, oob_score=True )
    clf = clf.fit(X, y)
    out_of_bag_errors.append(1 - clf.oob_score_)
    
print "Time elapsed for OOB: " , datetime.datetime.now() - start

plt.plot(estimators_nums , out_of_bag_errors)
plt.title('Number of trees in the random forest vs OOB error')
plt.xlabel('Number of trees in the Random Forest ')
plt.ylabel('OOB error')
plt.show()


#10-fold CV miss-classification error for RandomForest of different number of trees

start = datetime.datetime.now()
estimators_nums = [ 1,2,4,8,16,32,64,128]
CV_errors = []
 
for j , est_num in enumerate(estimators_nums):
    print est_num
    clf = RandomForestClassifier(n_estimators=est_num )
    clf = clf.fit(X, y.target.values)
    scores = cross_val_score(clf, X, y.target.values, cv=10)
    CV_errors.append( 1- scores.mean())
print "Time elapsed for CV curve: " , datetime.datetime.now() - start     

plt.plot(estimators_nums , CV_errors)
plt.title('Number of trees in the random forest vs 10-fold CV')
plt.xlabel('Number of trees in the Random Forest ')
plt.ylabel('CV error rate')
plt.show()


#10-fold CV AUC for RandomForest of different number of trees

start = datetime.datetime.now()
estimators_nums = [ 1,2,4,8,16,32,64,128]
CV_aucs = []
 
for j , est_num in enumerate(estimators_nums):
    print est_num
    clf = RandomForestClassifier(n_estimators=est_num )
    clf = clf.fit(X, y.target.values)
    auc = cross_val_score(clf, X, y.target.values, cv=10 , scoring='roc_auc')
    CV_aucs.append( auc.mean())
print "Time elapsed for CV curve: " , datetime.datetime.now() - start     

plt.plot(estimators_nums , CV_aucs)
plt.title('Number of trees in the random forest vs 10-fold CV AUC')
plt.xlabel('Number of trees in the Random Forest ')
plt.ylabel('ROC AUC')
plt.show()


#Bias-Variance analysis for RandomForest of different number of trees

estimators_nums = [ 1,2,4,8,16,32,64,128]
bias_values = []
variance_values = []
for e in estimators_nums:
    var_bias_df = DataFrame(y , columns = ['target' ] ) 
    for i in range(bootstrap_nums):
        training_samples,  training_labels = resample(X, y)
        test_index = diff(X.index , training_samples.index.unique() )
        test_samples = X.iloc[test_index,:]
        test_labels =  y.iloc[test_index] 
        clf = RandomForestClassifier(n_estimators=e)
        clf.fit(training_samples, training_labels)
        preds = clf.predict(test_samples)
        booster_preds = DataFrame( preds , index=test_samples.index , columns = ['booster_' + str(i) ]) 
        var_bias_df = var_bias_df.join(booster_preds)
        
    print "=====Bootstrapping done======"  
    var_bias_df['bias'] = var_bias_df.apply(  loss_func ,  axis = 1)
    var_bias_df['var'] = var_bias_df.apply(  variance_func ,  axis = 1)
    print var_bias_df
    
    bias_list = var_bias_df['bias'].values.tolist()
    var_list = var_bias_df['var'].values.tolist()
    print np.mean(list(filter(lambda x: x!= -1  , bias_list)))
    bias_values.append( np.mean(list(filter(lambda x: x!= -1  , bias_list))))
    variance_values.append( np.mean(list(filter(lambda x: x!= -1, var_list))))
    print "==========="

plt.figure(1)
plt.subplot(211)
plt.plot(estimators_nums, bias_values)
plt.xticks(estimators_nums)
plt.xlabel('Trees number in the RandomForest')
plt.ylabel('bias')

plt.subplot(212)
plt.plot(estimators_nums, variance_values)
plt.xticks(estimators_nums)
plt.xlabel('Trees number in the RandomForest')
plt.ylabel('Variance')
plt.show()







