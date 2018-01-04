#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 14:18:15 2018

@author: stefan
"""
import tensorflow as tf
def mmodel(images,batch_size):
    with tf.variable_scope('conv1_1') as scope:
        weights = tf.get_variable('weights', 
                                  shape = [3,3,3, 64],
                                  dtype = tf.float32, 
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable('biases', 
                                 shape=[64],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1_1 = tf.nn.relu(pre_activation, name= scope.name)
    with tf.variable_scope('conv1_2') as scope:
        weights = tf.get_variable('weights', 
                                  shape = [3,3,64, 64],
                                  dtype = tf.float32, 
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable('biases', 
                                 shape=[64],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(conv1_1, weights, strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1_2 = tf.nn.relu(pre_activation, name= scope.name)
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1,2,2,1],strides=[1,2,2,1],
                               padding='SAME', name='pooling1')
        #norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
        #                  beta=0.75,name='norm1')
        
        
    with tf.variable_scope('conv2_1') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,64,128],
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable('biases',
                                 shape=[128], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool1, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2_1 = tf.nn.relu(pre_activation, name='conv2_1')
    with tf.variable_scope('conv2_2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,128,128],
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable('biases',
                                 shape=[128], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(conv2_1, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2_2 = tf.nn.relu(pre_activation, name='conv2_2') 
    with tf.variable_scope('pooling2_lrn') as scope:
        #norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          #beta=0.75,name='norm2')
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1,2,2,1], strides=[1,2,2,1],
                               padding='SAME',name='pooling2')
        
        
    with tf.variable_scope('conv3_1') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,128,256],
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable('biases',
                                 shape=[256], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool2, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3_1 = tf.nn.relu(pre_activation, name='conv3_1')
    with tf.variable_scope('conv3_2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,256,256],
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable('biases',
                                 shape=[256], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(conv3_1, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3_2 = tf.nn.relu(pre_activation, name='conv3_2')
    with tf.variable_scope('conv3_3') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,256,256],
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable('biases',
                                 shape=[256], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(conv3_2, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3_3 = tf.nn.relu(pre_activation, name='conv3_3')
    with tf.variable_scope('pooling3_lrn') as scope:
        #norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          #beta=0.75,name='norm2')
        pool3 = tf.nn.max_pool(conv3_3, ksize=[1,2,2,1], strides=[1,2,2,1],
                               padding='SAME',name='pooling3')
    
    
    with tf.variable_scope('conv4_1') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,256,512],
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable('biases',
                                 shape=[512], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool3, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4_1 = tf.nn.relu(pre_activation, name='conv4_1')
    with tf.variable_scope('conv4_2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,512,512],
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable('biases',
                                 shape=[512], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(conv4_1, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4_2 = tf.nn.relu(pre_activation, name='conv4_2')
    with tf.variable_scope('conv4_3') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,512,512],
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable('biases',
                                 shape=[512], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(conv4_2, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4_3 = tf.nn.relu(pre_activation, name='conv4_3')
    with tf.variable_scope('pooling4_lrn') as scope:
        #norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          #beta=0.75,name='norm2')
        pool4 = tf.nn.max_pool(conv4_3, ksize=[1,2,2,1], strides=[1,2,2,1],
                               padding='SAME',name='pooling4')
    
    
    with tf.variable_scope('conv5_1') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,512,512],
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable('biases',
                                 shape=[512], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool4, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv5_1 = tf.nn.relu(pre_activation, name='conv5_1')
    with tf.variable_scope('conv5_2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,512,512],
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable('biases',
                                 shape=[512], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(conv5_1, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv5_2 = tf.nn.relu(pre_activation, name='conv5_2')
    with tf.variable_scope('conv5_3') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,512,512],
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable('biases',
                                 shape=[512], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(conv5_2, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv5_3 = tf.nn.relu(pre_activation, name='conv5_3')
    with tf.variable_scope('pooling5_lrn') as scope:
        #norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          #beta=0.75,name='norm2')
        pool5 = tf.nn.max_pool(conv5_3, ksize=[1,2,2,1], strides=[1,2,2,1],
                               padding='SAME',name='pooling5')
    
    
    with tf.variable_scope('fc6') as scope:
        reshape = tf.reshape(pool5, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim,4096],
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('biases',
                                 shape=[4096],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.1))
        fc6 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        fc6_drop = tf.nn.dropout(fc6, 0.5, name = "fc6_drop")
        
    with tf.variable_scope('fc7') as scope:
        
        dim = fc6_drop.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim,4096],
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('biases',
                                 shape=[4096],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.1))
        fc7 = tf.nn.relu(tf.matmul(fc6_drop, weights) + biases, name=scope.name)
        fc7_drop = tf.nn.dropout(fc7, 0.5, name = "fc7_drop")
    
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[4096, 5],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases', 
                                 shape=[5],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(fc7_drop, weights), biases, name='softmax_linear')
    return softmax_linear

def loss(logits,label_batches):
     cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=label_batches)
     cost = tf.reduce_mean(cross_entropy)
     #add loss in summary
     tf.summary.scalar("loss", cost)
     return cost

def get_accuracy(logits, labels):
    acc = tf.nn.in_top_k(logits, labels , 1)
    acc = tf.cast(acc, tf.float32)
    acc = tf.reduce_mean(acc)
    #add acc in summary
    tf.summary.scalar("acc", acc)
    return acc

def training(loss, lr):
    train_op = tf.train.RMSPropOptimizer(lr, 0.9).minimize(loss)
    return train_op
