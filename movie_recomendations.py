from pymongo import MongoClient
import os
from dotenv import load_dotenv
import requests
from sentence_transformers import SentenceTransformer
load_dotenv()

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
   
def embed (sentence):
    embeddings = model.encode(sentence)
    return embeddings
def generate_embedings(payload):
    API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/sentence-similarity"
    headers = {
    "Authorization": f"Bearer {os.getenv('HF_TOKEN')}",
    }
    
    print (headers)
    response = requests.post(API_URL,
                             headers=headers,
                             json =payload
                             )
    if response.status_code!=200:
        raise ValueError(f"request failed:{response.text}")
    return response.json()

obj =  {
            "inputs": {
            "source_sentence": "That is a happy person",
            "sentences": [
                "That is a happy dog",
                "That is a very happy person",
                "Today is a sunny day"
                    ]
                 },
            }
print(embed(["This is an example sentence", "Each sentence is converted"]))

def connect_mongo():
    mongo_url = os.getenv("MONGO_URL")
    print(mongo_url)
    client = MongoClient(mongo_url) 
    return client
m_client = connect_mongo()
db = m_client["sample_mflix"]
collection = db['movies']


for doc in collection.find({'plot':{'$exists':True}}):
    content = []
    content.append(doc["plot"])
    
    doc["plot_embeding_hf"] = embed(content).flatten().tolist()
   
    collection.replace_one({'_id':doc["_id"]},doc)

