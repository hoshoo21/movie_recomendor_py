from app.movie_recomendations import embed, connect_mongo


m_client = connect_mongo()
db = m_client["sample_mflix"]
collection = db["movies"]
pipeline = [
    {
        "$vectorSearch": {
            "index": "default",
            "path": "plot_embeding_hf",
            "queryVector":embed("communism").flatten().tolist(), # Ensure this is a flat list of floats
            "numCandidates": 100,
            "limit": 5
        }
    }
]
results = collection.aggregate(pipeline)
for document in results:
    print (f'Movie Name: {document["title"] }\n Movie Plot: {document["plot"]} ')


