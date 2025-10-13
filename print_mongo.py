from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client['casacom']
collection = db['maids']

for doc in collection.find():
    print(doc)
    break