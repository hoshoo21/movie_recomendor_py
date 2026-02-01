from pymongo import MongoClient
import os
from dotenv import load_dotenv
import requests
from sentence_transformers import SentenceTransformer
import certifi
load_dotenv()

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
   
def embed (sentence):
    embeddings = model.encode(sentence)
    return embeddings

            

def connect_mongo():
    try:
        mongo_url = os.getenv("MONGO_URL")
        client = MongoClient(
            mongo_url,
            serverSelectionTimeoutMS=60000, 
            connectTimeoutMS=60000,
            socketTimeoutMS=60000,
            tlsCAFile=certifi.where(),
            tlsAllowInvalidCertificates=True) 
        print ("mongo db connected")
        client.admin.command('ping')
        print("Finally Connected!")
        return client
    except:
        print ("Mongo db connectivity issue")
m_client = connect_mongo()
db = m_client["sample_mflix"]
collection = db['movies']

query = {
    "$and": [
        {"plot": {"$exists": True}},
        {"plot_embeding_hf": {"$exists": False}}
    ]
}
try :
    items = collection.find(query)
except:
    print ("error fetching records")
#items = collection.find(query).limit(100)
count = 0
for doc in items:
    content = []
    content.append(doc["plot"])
    doc["plot_embeding_hf"] = embed(content).flatten().tolist()
   
    collection.replace_one({'_id':doc["_id"]},doc)
    count = count+1
   