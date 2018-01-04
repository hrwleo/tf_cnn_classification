#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 14:18:15 2018

@author: stefan
"""
import tensorflow as tf
import numpy as np
import read_image as inputData
import model
import os
from datetime import datetime


def run_training():
    data_dir = './flower_photos/'
    log_dir = './log/'
    image, label = inputData.get_files(data_dir)
    image_batches, label_batches = inputData.get_batches(image,label,224,224,16,20)
    p = model.mmodel(image_batches, 16)
    cost = model.loss(p,label_batches)
    train_op = model.training(cost,0.001)
    acc = model.get_accuracy(p,label_batches)
    
    sess = tf.Session()
    init = tf.global_variables_initializer()
    #merge all summary
    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir, sess.graph)
    
    sess.run(init)
    saver = tf.train.Saver()
   
    
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess,coord = coord)
    
    try:
       for step in np.arange(1000):
           if coord.should_stop():
               break
           _,train_acc,train_loss = sess.run([train_op,acc,cost])
           print("{} step:{} loss:{} accuracy:{}"
                 .format(datetime.now(),step,train_loss,train_acc))
           if step % 250 == 0:
               #record the summary
               summary = sess.run(summary_op)
               train_writer.add_summary(summary, step)
               
               check = os.path.join(log_dir, "mmodel.ckpt")
               saver.save(sess, check, global_step=step)
    except tf.errors.OutOfRangeError:
        print("Done!!!")
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()
    
if __name__ == "__main__":
    run_training()
    print("train done")
    
    
    