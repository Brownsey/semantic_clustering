import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

txt_data = pd.read_csv('data/output.txt', sep='\t', header= None, names = ["text"])
sentences = txt_data['text'].tolist()

model_ckpt = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)

encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    model_output = model(**encoded_input)
    
    
token_embeddings = model_output.last_hidden_state
print(f"Token embeddings shape: {token_embeddings.size()}")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
# Normalize the embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
print(f"Sentence embeddings shape: {sentence_embeddings.size()}")


sentence_embeddings = sentence_embeddings.detach().numpy()

scores = np.zeros((sentence_embeddings.shape[0], sentence_embeddings.shape[0]))

for idx in range(sentence_embeddings.shape[0]):
    scores[idx, :] = cosine_similarity([sentence_embeddings[idx]], sentence_embeddings)[0]

scores



# if we find sentences that are similiar
# remove them the overall list based on a given threshold
# eventually, just return the input dataframe with the ids_to_remove removed
# in theory this should give us a list of unique sentences



def get_matches(scores, threshold):
    matches = [i for i, v in enumerate(scores) if v > threshold]
    return matches, len(matches)
    

def run_matcher(data, threshold):
    ids_to_remove = [] # Ids of the sentences that are similar and can be removed
    ids_to_return = [] # Ids of the original similar sentences to be returned
    for idx in range(0, len(data)):
        if idx in ids_to_remove:
            continue
        matches, length = get_matches(scores[idx], threshold)
        if(length > 1):
            ids_to_return.append(matches[0])
            ids_to_remove.extend(matches[1:])
        else:
            ids_to_return.append(matches[0])
    return ids_to_return, ids_to_remove

ids_to_return, ids_to_remove = run_matcher(scores, 0.7)
txt_data.iloc[ids_to_return]
#Define the threshold and get the indices of the sentences that are similar
