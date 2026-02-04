import numpy as np 
import pandas as pd 
from sentence_transformers import SentenceTransformer
from sklearn.metrics import DistanceMetric
from sklearn.decomposition import PCA
import  matplotlib.pyplot as plt 
import matplotlib as mpl
from sklearn.ensemble import RandomForestClassifier 



def load_model(model_name):
    #all-MiniLM-L6-V2

    return SentenceTransformer(model_name)

    
def encode(model,texts):
    return model.encode(texts)
    
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

def get_query_results(model,query):
    encode(model,query)
    
def generate_scores (raw_data, embeddings):
    
    raw_data = raw_data.reset_index(drop=True)
    embeddings = embeddings.reset_index(drop=True)
    embeddings['target_role'] = raw_data['role']
    role_counts = embeddings['target_role'].value_counts()
    rare_roles = role_counts[role_counts < 3].index
    embeddings['target_role'] = embeddings['target_role'].replace(rare_roles, 'Other')    
    X = embeddings.iloc[:,:-1]
    y = embeddings.iloc[:,-1]

    ctf = RandomForestClassifier(max_depth=2, random_state=0).fit(X,y)
    pca = PCA(n_components=2).fit(X)
    
def get_distance (embeddings, query_embeddings):
    dist = DistanceMetric.get_metric('euclidean')
    dist_arr =dist.pairwise(embeddings, query_embeddings.reshape(1,-1)).flatten()
    print(dist_arr)
    idist_arr_sorted = dist_arr.argsort()
    print(idist_arr_sorted)
    return idist_arr_sorted