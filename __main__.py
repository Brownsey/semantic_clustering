from classes.semantic_class import SemanticClass
import pandas as pd
import time


def main():
    start = time.time()
    txt_data = pd.read_csv('data/output.txt', sep='\t', header= None, names = ["text"])
    sc = SemanticClass(txt_data, threshold= 0.7)
    df = sc.main()
    df.to_csv('data/unique_sentences.txt', sep='\t', header=False, index=False)
    end = time.time()
    print("The time taken to run the script is " + str(round((end - start), 1)) + " seconds")
    return df

if __name__ == "__main__":
    main()
