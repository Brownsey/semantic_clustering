import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SemanticClass:

    def __init__(self, df: pd.DataFrame, threshold: float = 0.7, text_column: str = "text"):
        self.df = df                                    # input df
        self.threshold = threshold                      # threshold for similarity
        self.text_column = text_column                  # column name for text
        self.__convert_to_list()                        # convert df to list if applicable

    def __convert_to_list(self):
        self.list_df = self.df[self.text_column].tolist()

    @staticmethod
    def __mean_pooling(model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def __get_scores(self):
        sentences = self.list_df
        model_ckpt = "sentence-transformers/all-MiniLM-L6-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        model = AutoModel.from_pretrained(model_ckpt)

        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            model_output = model(**encoded_input)
            
            
        token_embeddings = model_output.last_hidden_state
        print(f"Token embeddings shape: {token_embeddings.size()}")

        sentence_embeddings = self.__mean_pooling(model_output, encoded_input["attention_mask"])

        # Normalize the embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        print(f"Sentence embeddings shape: {sentence_embeddings.size()}")


        sentence_embeddings = sentence_embeddings.detach().numpy()

        scores = np.zeros((sentence_embeddings.shape[0], sentence_embeddings.shape[0]))

        for idx in range(sentence_embeddings.shape[0]):
            scores[idx, :] = cosine_similarity([sentence_embeddings[idx]], sentence_embeddings)[0]
        self.scores = scores
        return scores
        
    @staticmethod
    def __get_matches(scores, threshold):
        matches = [i for i, v in enumerate(scores) if v > threshold]
        return matches, len(matches)
        

    def __run_matcher(self):
        data = self.list_df
        threshold = self.threshold
        scores = self.scores
        ids_to_remove = [] # Ids of the sentences that are similar and can be removed
        ids_to_return = [] # Ids of the original similar sentences to be returned
        for idx in range(0, len(data)):
            if idx in ids_to_remove:
                continue
            matches, length = self.__get_matches(scores[idx], threshold)
            if(length > 1):
                ids_to_return.append(matches[0])
                ids_to_remove.extend(matches[1:])
            else:
                ids_to_return.append(matches[0])
        self.ids_to_return = ids_to_return
        return ids_to_return, ids_to_remove

    def __return_df(self):
        return self.df.iloc[self.ids_to_return]


    def main(self):
        self.__get_scores()
        self.__run_matcher()
        return self.__return_df()


    
#Define the threshold and get the indices of the sentences that are similar
