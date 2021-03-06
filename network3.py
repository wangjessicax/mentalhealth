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
    totalList2=list()
    depList2 =list()
    ageList2=list()
    for x in range(0, size):
        totalList.append(contents[x*2033:(x+1)*2033])
        depList.append(contents[(x*2033)+113]) #113 question
        ageList.append(contents[(x*2033)+121:((x+1)*2033)+122])

    #cycle through string and convert into array
    
    for s in totalList:
        charList = list()

        try:
            for c in s:
                if c == ".":
                    charList.append(10)
                elif c.isspace() is True:  # if string is empty
                    charList.append(10)
                else:
                    char=int(c)
                    charList.append(char)
            totalList2.append(np.asarray(charList))
        except:
            charList.append(10)
            totalList2.append(np.asarray(charList))
       

    #cycle through dep list and convert into array
    for s in depList:
        charList = np.zeros(10)

        try:
            for c in s:
                if c == ".":
                    charList[9]=1
                elif c.isspace() is True:  # if string is empty
                    charList[9]=1
                else:
                    char=int(c)
                    charList[char]=1
            depList2.append(np.asarray(charList))

        except:
            charList[9]=1
            depList2.append(np.asarray(charList))
            

    for s in ageList:
        charList = list()

        try:
            for c in s:
                if c == ".":
                    charList.append(10)
                elif c.isspace() is True:  # if string is empty
                    charList.append(10)
                else:
                    char=int(c)
                    charList.append(char)
            ageList2.append(np.asarray(charList))
        except:
            charList.append(10)
            ageList2.append(np.asarray(charList))

    
    #flatten the arrays within the array

    totalList_array=np.asarray(totalList2)
    depList_array=np.asarray(depList2)
    ageList_array=np.asarray(ageList)

    #Prepend the column of 1s for bias
    print(totalList_array.shape)
    N=totalList_array.shape[0]
    M=totalList_array.shape[1]
    print("Total list...:")
    print(totalList_array[0].shape)
    totallist_array=totalList_array.flatten()
    #all_X = np.ones((N, M + 1))
    all_X = np.reshape(totalList_array, (size,2033))


    # Convert into one-hot vectors
    num_labels = max(np.unique(depList_array))+1

    all_Y = depList_array

   # print(all_Y)
    return train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED)

def main():
    train_X, test_X, train_y, test_y = get_data()

    # Layer's sizes
    x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
    h_size = 256                # Number of hidden nodes
    y_size = train_y.shape[1]   # Number of outcomes (3 iris flowers)
    print(x_size)
    print(y_size)
    print(train_X.shape)
    print(train_y.shape)
    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    # Forward propagation
    yhat    = forwardprop(X, w_1, w_2)
    print(yhat.shape)
    predict = tf.argmax(yhat, axis=1)

    # Backward propagation
    #yhat = tf.transpose(yhat)
    cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    fh = open("brainresults.txt","w")
    saver = tf.train.Saver()
    for epoch in range(10000):
        # Train with each example
        for i in range(len(train_X)):
            sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})

        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: train_X, y: train_y}))
        test_accuracy  = np.mean(np.argmax(test_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: test_X, y: test_y}))

        save_path = saver.save(sess, "./model.ckpt")

        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
              % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

        fh.write("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
              % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))


    fh.close()
    sess.close()
'''
def test():

    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, "/tmp/model.ckpt")

    #sess.run(predict, feed_dict={X: , y: test_y})
    
'''

if __name__ == '__main__':
    size = 10000 #60346
    main()