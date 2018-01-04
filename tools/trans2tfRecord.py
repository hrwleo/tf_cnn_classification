#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 16:03:39 2018

@author: stefan
"""
import numpy as np
import cv2
import tensorflow as tf
import os


def load_file(example_list_file):
    lines = np.genfromtxt(example_list_file, delimiter=" ", 
                          dtype=[('col1', 'S120'), ('col2', 'i8')])
    examples = []
    labels = []
    for example,label in lines:
        examples.append(example)
        labels.append(label)
        
    #convert to numpy array
    return np.asarray(examples), np.asarray(labels), len(lines)

def extract_image(filename, height, width):
    print(filename)
    image = cv2.imread(filename)
    image = cv2.resize(image, (height, width))
    b,g,r = cv2.split(image)
    rgb_image = cv2.merge([r,g,b])
    return rgb_image

def trans2tfRecord(train_file,name,output_dir,height,width):
    if not os.path.exists(output_dir) or os.path.isfile(output_dir):
        os.makedirs(output_dir)
    _examples,_labels,examples_num = load_file(train_file)
    filename = name + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(filename)
    for i,[example,label] in enumerate(zip(_examples,_labels)):
        print("NO{}".format(i))
        #need to convert the example(bytes) to utf-8
        example = example.decode("UTF-8")
        image = extract_image(example,height,width)
        image_raw = image.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw':_bytes_feature(image_raw),
                'height':_int64_feature(image.shape[0]),
                 'width': _int64_feature(32),  
                'depth': _int64_feature(32),  
                 'label': _int64_feature(label)                        
                }))
        writer.write(example.SerializeToString())
    writer.close()
    
def _int64_feature(value):  
     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))  
   
def _bytes_feature(value):  
     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    