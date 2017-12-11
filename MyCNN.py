import tensorflow as tf
import os
import numpy as np

class CNN(object):
    def __init__(self, image_dimensions, num_classes, filter_size, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None,image_dimensions[0],image_dimensions[1],image_dimensions[2]], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        l2_loss = tf.constant(0.0)
        with tf.name_scope("conv-maxpool-%s" % filter_size):
                
            # Convolution Layer 1
            filter_shape = [filter_size, filter_size, 1, 16] #out channel = 200(default)
            W1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W1")
            b1 = tf.Variable(tf.constant(0.1, shape=[16]), name="b1")
            conv1 = tf.nn.conv2d(self.input_x,W1,strides=[1, 1, 1, 1],padding="VALID",name="conv1")
                
            #Non linearity Layer1
            h1 = tf.nn.relu(tf.nn.bias_add(conv1, b1), name="relu1")
                
            #Pooling Layer 1
            pool1 = tf.nn.max_pool(h1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name="pool1")
                
            # Convolution Layer 2
            W2 = tf.Variable(tf.truncated_normal([filter_size,filter_size,16,32], stddev=0.1), name="W2")
            b2 = tf.Variable(tf.constant(0.1, shape=[32]), name="b2")
            conv2 = tf.nn.conv2d(pool1,W2,strides=[1, 1, 1, 1],padding="VALID",name="conv2")

            #Non linearity Layer2
            h2 = tf.nn.relu(tf.nn.bias_add(conv2, b2), name="relu2")
                
            #Pooling Layer 2
            pool2 = tf.nn.max_pool(h2, ksize=[1,2,2,1], strides=[1,1,1,1], padding='VALID', name="pool2")

            # Convolution Layer 3
            W3 = tf.Variable(tf.truncated_normal([filter_size,filter_size,32,64], stddev=0.1), name="W3")
            b3 = tf.Variable(tf.constant(0.1, shape=[64]), name="b3")
            conv3 = tf.nn.conv2d(pool2,W3,strides=[1, 2, 2, 1],padding="VALID",name="conv3")

            #Non linearity Layer3
            h3 = tf.nn.relu(tf.nn.bias_add(conv3, b3), name="relu3")
                
            #Pooling Layer 3
            pool3 = tf.nn.max_pool(h3, ksize=[1,2,2,1], strides=[1,1,1,1], padding='VALID', name="pool3")
            
            # Convolution Layer 4
            W4 = tf.Variable(tf.truncated_normal([filter_size,filter_size,64,128], stddev=0.1), name="W4")
            b4 = tf.Variable(tf.constant(0.1, shape=[128]), name="b4")
            conv4 = tf.nn.conv2d(pool3,W4,strides=[1, 2, 2, 1],padding="VALID",name="conv4")

            #Non linearity Layer4
            h4 = tf.nn.relu(tf.nn.bias_add(conv4, b4), name="relu4")
                
            #Pooling Layer 4
            pool4 = tf.nn.max_pool(h4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name="pool4")


            print(pool4.shape)
            #Fully connected layer
            fc = tf.reshape(pool4,[-1,2048])
            Wfc = tf.Variable(tf.truncated_normal([2048,512],stddev=0.1))
            bfc =   tf.Variable(tf.constant(0.1, shape=[512]), name="bfc")
            out = tf.matmul(fc,Wfc)+bfc

            #Fully connected layer2
            
            Wfc2 = tf.Variable(tf.truncated_normal([512,128],stddev=0.1))
            bfc2 =   tf.Variable(tf.constant(0.1, shape=[128]), name="bfc2")
            out2 = tf.matmul(out,Wfc2)+bfc2

        #with tf.name_scope("dropout"):
        #    self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([128,num_classes],stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(out2, W, b, name="scores")
            self.softmax_scores = tf.nn.softmax(self.scores,name="softmax_outputs")
            print(self.scores.shape)
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            print(self.predictions.shape)

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
