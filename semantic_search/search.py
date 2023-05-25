import numpy as np
import torch
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import os
import faiss
import time


os.environ['KMP_DUPLICATE_LIB_OK']="True"

data=pd.read_csv('data.csv')
# Load a pre-trained model
model =SentenceTransformer('msmarco-MiniLM-L-12-v3')
news=data["content"].tolist()
print(news)

# Embedding news using SentenceTransformer
news_embeeding=model.encode(news)
print(news_embeeding)

# Indexing using FAISS
index = faiss.IndexFlatL2(news_embeeding.shape[1])
index.add(news_embeeding)
faiss.write_index(index, 'news')
index = faiss.read_index('news')

print(index)

def search(query):
    
    t=time.time()
    query_vector = model.encode([query])
    k = 5
    top_k = index.search(query_vector, k)
    print('totaltime: {}'.format(time.time()-t))
    return [news[_id] for _id in top_k[1].tolist()[0]]

results=search("台積電")
print(results)

##Ref : https://medium.com/mlearning-ai/how-to-build-a-semantic-search-engine-using-python-5c68e8442df1