# Implementation of a simple MLP network with one hidden layer. Tested on the iris data set.
# Requires: numpy, sklearn>=0.18.1, tensorflow>=1.0

# NOTE: In order to make the code simple, we rewrite x * W_1 + b_1 = x' * W_1'
# where x' = [x | 1] and W_1' is the matrix W_1 appended with a new row with elements b_1's.
# Similarly, for h * W_2 + b_2
import tensorflow as tf
import numpy as np
from numpy import array
from sklearn.model_selection import train_test_split
import os
import sys
import fileinput

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)


def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
    yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat


def get_data():

    """ Read the iris data set and split them into training and test sets """
    file = open("data.txt","r") 
    contents=file.read()
        
    #get all of X and Y data
    totalList=list()
    depList = list()
    ageList = list()
    for x in range(0, 60346):
        totalList.append(contents[x*2033:(x+1)*2033])
        depList.append(contents[(x*2033)+113]) #113 question
        ageList.append(contents[(x*2033)+121:((x+1)*2033)+122])


    #cycle through string and convert into array
    
    for s in totalList:
        print("this is the string")
        print(s)
        charList = list()
        for c in s:
            if c.isspace() is true:  # if string is empty
                charList.append(-1)
            if c:
                char=float(c)
                charList.append(char)
        print(charList)
        totalList2.append(charList)

    print(totalList2)



    #convert to numpy array     
    totalList_array=np.asarray(totalList)
    depList_array=np.asarray(depList)
    ageList_array=np.asarray(ageList)
    '''
    #cycle through numpy array and convert from string into numpy array
    for x in np.nditer(totalList_array):
        print(type(x))
        x=x.replace(" ", ".")
        print(x)
        x=np.fromstring(x, dtype=int, sep='')
        print(x)
    '''

    #Prepend the column of 1s for bias
    print(totalList_array.shape)
    N = totalList_array.shape[0]
    M=1
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = np.reshape(totalList_array,(60346,1))

    s = '5.2 5.6 5.3'
    floats = [float(x) for x in s.split()]

    # Convert into one-hot vectors
    num_labels = len(np.unique(depList_array))
    all_Y = np.eye(num_labels)[depList_array]  # One linher trick!
    return train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED)

def main():
    train_X, test_X, train_y, test_y = get_data()

    # Layer's sizes
    x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
    h_size = 256                # Number of hidden nodes
    y_size = train_y.shape[1]   # Number of outcomes (3 iris flowers)

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    # Forward propagation
    yhat    = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, axis=1)

    # Backward propagation
    cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(100):
        # Train with each example
        for i in range(len(train_X)):
            sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})

        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: train_X, y: train_y}))
        test_accuracy  = np.mean(np.argmax(test_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: test_X, y: test_y}))

        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
              % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

    sess.close()

if __name__ == '__main__':
    main()