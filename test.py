import tensorflow as tf
from openml.apiconnector import APIConnector
import numpy as np
from pandas import Series,DataFrame
import pandas as pd
import datetime



from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)



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

print X_train.shape
print Y_train.shape
print X_test.shape
print Y_test.shape


Y_tr = np.zeros((len(Y_train) , 10 ))
Y_tr[ np.arange(len(Y_train))  , Y_train.astype(int) ] = 1

Y_te = np.zeros((len(Y_test) , 10 ))
Y_te[ np.arange(len(Y_test))  , Y_test.astype(int) ] = 1


print Y_train[10]
print Y_tr[10]


x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)





mini_batch_size =50
epoch = 1000

k=0
for p in np.arange(0,epoch):
    #start = datetime.datetime.now()
    #for k in xrange(0, len(X_train), mini_batch_size):
    batch_xs = X_train[k:k+mini_batch_size]
    batch_ys = Y_tr[k:k+mini_batch_size]
        #batch_xs , batch_ys = mnist.train[k:k+mini_batch_size]
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    k += mini_batch_size
    if k >= X_train.size - mini_batch_size:
        k=0
        
    #print datetime.datetime.now() - start 

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: X_test, y_: Y_te}) )











#----------------------------------------------------------- mini_batch_size =50
#------------------------------------------------------------------ epoch = 1000
#------------------------------------------------------------------------------ 
#-------------------------------------------------- for p in np.arange(0,epoch):
    #------------------------------------------- start = datetime.datetime.now()
    #------------------------ for k in xrange(0, len(X_train), mini_batch_size):
        #------------------------------- batch_xs = X_train[k:k+mini_batch_size]
        #---------------------------------- batch_ys = Y_tr[k:k+mini_batch_size]
        #--------------- #batch_xs , batch_ys = mnist.train[k:k+mini_batch_size]
        #----------- sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    #------------------------------------- print datetime.datetime.now() - start
#------------------------------------------------------------------------------ 
#---------------- correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#------------ accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#------------------- print(sess.run(accuracy, feed_dict={x: X_test, y_: Y_te}) )

