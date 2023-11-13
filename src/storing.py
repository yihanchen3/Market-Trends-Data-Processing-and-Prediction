"""This module implements a CRUD (Create, Read, Update, Delete) interface to our noSQL database."""

import os
import pymongo

myclient = pymongo.MongoClient(
    "mongodb+srv://Cyhan103:yih1900Ss@daps2022.l7mstiw.mongodb.net/?retryWrites=true&w=majority")


def store_cloud(stock_data, external_data):
    db = myclient.daps2022
    collection = db.stock
    collection.drop()
    collection.insert_many(stock_data.to_dict('records'))

    collection = db.covid
    collection.drop()
    collection.insert_many(external_data.to_dict('records'))
    print('data stored in mongodb')


def store_local(stock_data, external_data, stock_name, external_name):
    stock_data.to_csv("./data/%s.csv" % stock_name)
    external_data.to_csv("./data/%s.csv" % external_name)
    print('data stored locally in the folder')
