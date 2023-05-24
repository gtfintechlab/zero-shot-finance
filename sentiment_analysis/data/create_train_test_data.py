import pandas as pd
from sklearn.model_selection import train_test_split

def encode(label_word):
    if label_word == "positive":
        return 0
    elif label_word == "negative":
        return 1
    elif label_word == "neutral":
        return 2
    else: 
        return -1

df = pd.read_csv("./ExtractedFinancialPhraseBank/Sentences_AllAgree.txt", sep="@", encoding ='latin1', names=['sentence', 'label'])

df["label"] = df["label"].apply(lambda x: encode(x))

for seed in [5768, 78516, 944601]: 
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)
    train_df.to_excel(f'./train/FPB-sentiment-analysis-allagree-train-{seed}.xlsx', index=False)
    test_df.to_excel(f'./test/FPB-sentiment-analysis-allagree-test-{seed}.xlsx', index=False)