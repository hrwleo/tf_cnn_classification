#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 14:18:15 2018

@author: stefan
"""
import Image
import tensorflow as tf
import numpy as np
import model

def get_one_image(img_dir):
    image = Image.open(img_dir)
    image = image.resize([32, 32])
    image_arr = np.array(image)
    return image_arr

def test(test_file):
    log_dir = './log/'
    image_arr = get_one_image(test_file)
    
    with tf.Graph().as_default():
        image = tf.cast(image_arr, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1,32, 32, 3])
        
        p = model.mmodel(image,1)
        logits = tf.nn.softmax(p)
        x = tf.placeholder(tf.float32,shape = [32,32,3])
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success')
            else:
                print('No checkpoint')
            prediction = sess.run(logits, feed_dict={x: image_arr})
            max_index = np.argmax(prediction)
            print(max_index)
            
if __name__ == "__main__":
    test_file = "test.jpeg"
    test(test_file)
    print("prediction over!")
    
    
    
    
    
    
    