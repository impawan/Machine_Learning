# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 10:55:08 2018

@author: paprasad
"""
import os
import csv

def config_manager(name):
    config_file = open("config.config",'r+')
    lines = config_file.readlines()
    TempDict = {}
    
    if (len(lines) > 0):
        for line in lines:
            print(line)
            key,username = line.split(":")
            TempDict[key] = username   
        if (name in str(TempDict.values()).strip()):
            print ("user face samples is already present in the face repositiroy")
            print ("please enter new user name")
            name = input()
            config_manager(name)
        else:
             max_key=max(TempDict, key=int)
             print (max_key)
             new_key = int(max_key)+1
             TempDict[new_key]=name
             temp = str(new_key)+":"+name
             config_file.write("\n"+str(temp))  
    else:
        config_file.write("1:"+name)
    config_file.close()
name = input()
config_manager(name)        
