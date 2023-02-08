import pandas as pd
txt_data = pd.read_csv('data/output.txt', sep='\t', header= None, names = ["text"])
text = txt_data['text'].tolist()

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(text)
print(embeddings)


#pooling for whole sentence, mean_pooling is average
#cosine similarity
