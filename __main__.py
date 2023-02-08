from classes.semantic_class import SemanticClass
import pandas as pd


def main():
    txt_data = pd.read_csv('data/output.txt', sep='\t', header= None, names = ["text"])
    sc = SemanticClass(txt_data)
    df = sc.main()
    df.to_csv('data/unique_sentences.txt', sep='\t', header=False, index=False)
    return sc.main()

if __name__ == "__main__":
    main()
