# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 10:55:08 2018

@author: paprasad
"""
import os
import csv

def config_manager(name):
    config_file = open("config.config",'r+')
    TempDict = {}
    if True:
        for row in config_file:
            key,username = row.split(':')
            TempDict[key] = username
        if (name in TempDict.values()):
            print ("user face samples is already present in the face repositiroy")
            print ("please enter new user name")
            name = input()
            config_manager(name)
        else:
             max_key=max(TempDict, key=int)
             new_key = int(max_key)+1
             TempDict[new_key]=name
             temp = str(new_key)+":"+name
             config_file.write("\n"+str(temp))  
            #config_file.truncate()
            #config_file.close()
    else:
        #TempDict[1]=name
        config_file.write("1:"+name)
#        #config_file.write(str(TempDict))
#        for key, value in TempDict.items():
#            #temp = int(key)+";"+str(TempDict[key])
#            temp = str(key).join(";".join(str(TempDict[key])))
#            config_file.write(str(temp))
    config_file.close()
name = input()
config_manager(name)        