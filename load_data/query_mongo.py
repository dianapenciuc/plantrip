
from pymongo import MongoClient
import pymongo
from pprint import pprint
import pandas as pd
from os import listdir
from os.path import isfile, join
import json

def load_files_from_dir(directory_path, collection):
    count = 0
    for dir in listdir(directory_path):
        dir_name = directory_path + "/" + dir
        filenames = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]
        count += len(filenames)
        
        for filename in filenames:
            with open(join(dir_name, filename)) as f:
                collection.insert_one(json.load(f))
    print(str(count)+ " documents inserés")


def load_data(directory_path, collection):
    for dir in listdir(directory_path):
        subdir_path = directory_path + dir
        load_files_from_dir(subdir_path, collection)


client = MongoClient(
    host="127.0.0.1",
    port = 27017,
    username = "admin",
    password = "pass"
)

print("Liste des bases de données :",client.list_database_names())
mydb = client["itineraire"]
print("La liste des collections :",mydb.list_collection_names())
pois = mydb["poi"]
#pprint(pois.find_one())
print("Nombre de documents :",pois.count_documents({}))

main_path = './datatourisme'
directory_path = main_path + '/nature_paris/objects/'

#load_data(directory_path,collection=pois)
print("Nombre de documents :",pois.count_documents({}))