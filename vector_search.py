from movie_recomendations import embed, connect_mongo


m_client = connect_mongo()
db = m_client["sample_mflix"]
collection = db["movies"]

results = collection.aggregate({
    {'$vectorSearch':{
        "queryVector":embed(["sex involved"]),
        "path":"plot_embeding_hf",
        "numCandidate":100,
        "limit" :4,
        "index" :"default"
        
    }}
})
for document in results:
    print (f'Movie Name: {document["title"] }\n Movie Plot: {document["plot"]} ')


