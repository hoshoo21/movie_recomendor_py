from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn
import pandas as pd 
import numpy as np 
import torch
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from semantic_search import load_model, encode, get_distance, generate_scores
from pathlib import Path
from search_classifier import classify_search_intent
from movie_recomendations import connect_mongo_compass, find_movies
from semantic_search import generate_embeddings
#model = SentenceTransformer(model_name, device="cpu" ,trust_remote_code=True)


df_resume= None
df_embeddings= None 
#load_model()
file_path = Path("./resumes/embeddings_train.csv")
df_resume = pd.read_csv('./resumes/resumes_train.csv')
    
if  file_path.exists()== False:
      embeddings = encode(df_resume['resume'].tolist())
      text_embedding_list = [embeddings[i] for i in range(len(embeddings))]
      column_names = ["embedding_" + str(i) for i in range(len(text_embedding_list[0]))]
      df_embeddings = pd.DataFrame(text_embedding_list, columns=column_names)
      df_embeddings.to_csv('resumes/embeddings_train.csv', index=False)
else :
     df_embeddings = pd.read_csv('./resumes/embeddings_train.csv') 

# generate_scores(df_resume, df_embeddings)



query = "I need to build infrastructure for my data"

#query_embeddings = encode(query)

#arr = get_distance(df_embeddings, query_embeddings)

# print(df_resume['role'].iloc[arr[:10]])
# print(df_resume.iloc[arr[:10]])
# text_embeddings = generate_embeddings(resumes_pd['resume'].tolist(), model)

# text_embedding_list = [text_embeddings[i] for i in range(len(text_embeddings))]
# column_names = ["embedding_" + str(i) for i in range(len(text_embedding_list[0]))]
# print (column_names)

# df_train = pd.DataFrame(text_embedding_list, columns=column_names)
# df_train['is_data_scientist'] = resumes_pd['role']=="Data Scientist"
# df_train.to_csv('resumes/embeddings_train.csv', index=False)
# print(df_train.head())

# X = df_train.iloc[:,:-1]
# y = df_train.iloc[:,-1]

# ctf = RandomForestClassifier(max_depth=2, random_state=0).fit(X,y)

# pca = PCA(n_components=2).fit(X)
# print(pca.explained_variance_ratio_)

# df_resume = pd.read_csv('resumes/resumes_test.csv')

# # generate embeddings
# text_embedding_list = generate_embeddings(df_resume['resume'].tolist(),model)
# text_embedding_list = [text_embedding_list[i] for i in range(len(text_embedding_list))]

# df_test = pd.DataFrame(text_embedding_list, columns=column_names)

# df_test['is_data_scientist'] = df_resume['role']=="Data Scientist"
# df_test.to_csv('resumes/embeddings_test.csv', index=False)
# df_test.head()

# X_test = df_test.iloc[:,:-1]
# y_test = df_test.iloc[:,-1]
# ctf.score(X_test,y_test)



db_client = None

@asynccontextmanager
async def lifespan(app:FastAPI):
    app.state.db_client = connect_mongo_compass()
    model_name ="Qwen/Qwen3-Embedding-0.6B"
    app.state.model = load_model(model_name=model_name)
    print(db_client)
    yield
    app.state.db_client = None

app = FastAPI(lifespan=lifespan)
class EmbedRequest(BaseModel):
    input: str

@app.post("/v1/embeddings")
async def embed(request: EmbedRequest):
    # Qwen3 usually requires the "query: " prefix for search tasks
    vector = model.encode(request.input, prompt="query: ").tolist()
    return {
        "data": [{"embedding": vector}],
        "model": "Qwen3-Embedding-0.6B",
          }

@app.post("/search")
async def handle_search(user_query):
    intent = classify_search_intent(user_query)
    if intent["strategy"] == "LEXICAL_HEAVY":
        pass
    else :
        movies =app.state.db_client.movies_coll.find({},{"plot_embeding_hf":1,"_id":1})
        embeddings_list = [
                    doc['plot_embedding'] 
                    async for doc in movies 
                    if 'plot_embedding' in doc
                ]
        allvect0r = np.array(embeddings_list)
        query_embeddings= await generate_embeddings(user_query,app.state.model)
        print (get_distance(allvect0r,query_embeddings))
   
if __name__ == "__main__":
    pass
   # uvicorn.run(app, host="0.0.0.0", port=8000)