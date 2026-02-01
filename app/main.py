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




def generate_embeddings(texts, model, batch_size=10):
    all_embeddings = []
    print(f"Starting embedding process for {len(texts)} items...")
    
    # Process in smaller batches (e.g., 10 resumes at a time)
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        print(f"Processing batch {i//batch_size + 1}...")
        
        # The actual math happens here
        batch_encodings = model.encode(batch)
        all_embeddings.extend(batch_encodings.tolist())
        
    print("All embeddings done!")
    return all_embeddings
model_name ="Qwen/Qwen3-Embedding-0.6B"
model = SentenceTransformer(model_name, device="cpu" ,trust_remote_code=True)

resumes_pd = pd.read_csv('./resumes/resumes_train.csv')


#embeddings = generate_embeddings(documents, model)

text_embeddings = generate_embeddings(resumes_pd['resume'].tolist(), model)

text_embedding_list = [text_embeddings[i] for i in range(len(text_embeddings))]
column_names = ["embedding_" + str(i) for i in range(len(text_embedding_list[0]))]
print (column_names)
print ("emedding done")

df_train = pd.DataFrame(text_embedding_list, columns=column_names)
df_train['is_data_scientist'] = resumes_pd['role']=="Data Scientist"
df_train.to_csv('resumes/embeddings_train.csv', index=False)
print(df_train.head())

X = df_train.iloc[:,:-1]
y = df_train.iloc[:,-1]
print (X,y)

ctf = RandomForestClassifier(max_depth=2, random_state=0).fit(X)
print (ctf.score(X,y))

pca = PCA(n_components=2).fit(X)
print(pca.explained_variance_ratio_)

df_resume = pd.read_csv('resumes/resumes_test.csv')

# generate embeddings
text_embedding_list = generate_embeddings(df_resume['resume'], my_sk)
text_embedding_list = [text_embedding_list[i].embedding for i in range(len(text_embedding_list))]

# store text embeddings in dataframe
df_test = pd.DataFrame(text_embedding_list, columns=column_names)

# create target variable
df_test['is_data_scientist'] = df_resume['role']=="Data Scientist"
df_test.to_csv('resumes/embeddings_test.csv', index=False)
df_test.head()

# define predictors and target
X_test = df_test.iloc[:,:-1]
y_test = df_test.iloc[:,-1]
ctf.score(X_test,y_test)

app = FastAPI()


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

if __name__ == "__main__":
    pass
   # uvicorn.run(app, host="0.0.0.0", port=8000)