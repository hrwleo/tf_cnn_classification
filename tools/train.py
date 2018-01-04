#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 16:25:49 2018

@author: stefan
"""

import tensorflow as tf
import numpy as np
import trans2tfRecord
import readtfRecord as rdData
import os
import model
from datetime import datetime


def run_training(file_tfRecord):
    log_dir = './log/'
    image, label = rdData.read_tfRecord(file_tfRecord)
    image_batches, label_batches = tf.train.batch([image, label],
                                                  batch_size=16, capacity = 20)
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
    name = "tfRecords"
    output_dir = "/home/stefan/Mycode/step_by_step/tools/tfRecordData/"
    trans2tfRecord.trans2tfRecord("train.txt", name, output_dir, 32, 32 )
    run_training("tfRecords.tfrecords")
    print("train done")
    
    
    