#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:55:55 2019

@author: lorajohns
"""
import numpy as np
from pprint import pprint
from pymongo import MongoClient
# import sys
def mongo_test():
# pprint library is used to make the output look more pretty
# connect to MongoDB, change the << MONGODB URL >> to reflect your own connection string
    client = MongoClient("mongodb://lora:loramongo@docdb-2019-07-24-22-00-09.cluster-c8bmqitihpyp.us-east-1.docdb.amazonaws.com:27017/?ssl=true&ssl_ca_certs=rds-combined-ca-bundle.pem&replicaSet=rs0")
    db=client.admin
# Issue the serverStatus command and print the results
    serverStatusResult=db.command("serverStatus")
    pprint(serverStatusResult)
    

list_of_arrays = np.array([[1,2],[3,4],[5,6],[7,8],[9,10]])

def unstack(list):
    return [x for y in list for x in y]
