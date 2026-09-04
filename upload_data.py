from pymongo.mongo_client import MongoClient
import pandas as pd
import json


from collections import UserList
# url

url = "mongodb+srv://pankaj:BYsKi7mFX6Kg8HKI@cluster0.okfhp97.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(url)


DATABASE_NAME = "pwskills"
COLLECTION_NAME = 'waferfault'

df = pd.read_csv("D:\NEW_AND_REPEAT_SENSOR_PROJECT_FOR_PEN_DRIVE\notebooks\wafer_23012020_041211 (1).csv")
df.head()

df = df.drop("Unnamed: 0" , axis = 1)
df

json_record = list(json.loads(df.T.to_json()).values())
json_record


type(json_record)

client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)