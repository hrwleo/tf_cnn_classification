#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 16:18:23 2018

@author: stefan
"""
import tensorflow as tf

def read_tfRecord(file_tfRecord):
    queue = tf.train.string_input_producer([file_tfRecord])
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(queue)
    features = tf.parse_single_example(
            serialized_example,
            features={
          'image_raw': tf.FixedLenFeature([], tf.string),  
          'height': tf.FixedLenFeature([], tf.int64), 
          'width':tf.FixedLenFeature([], tf.int64),
          'depth': tf.FixedLenFeature([], tf.int64),  
          'label': tf.FixedLenFeature([], tf.int64)  
                    }
            )
    image = tf.decode_raw(features['image_raw'],tf.uint8)
    #height = tf.cast(features['height'], tf.int64)
    #width = tf.cast(features['width'], tf.int64)
    image = tf.reshape(image,[32,32,3])
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    label = tf.cast(features['label'], tf.int64)
    print(image,label)
    return image,label
