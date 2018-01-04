#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 15:44:58 2018

@author: stefan
"""
import os

root_dir = "/home/stefan/Mycode/step_by_step"

def getTrainList():
    with open("train.txt", "w") as f:
        for file in os.listdir(root_dir + '/flower_photos'):
            for picFile in os.listdir(root_dir + '/flower_photos/' + file):
                f.write("flower_photos/"+file+"/"+picFile+" "+file+"\n")
                print(picFile)
            
if __name__ == "__main__":
    getTrainList()