import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from utils.results.decode import finer_ord_decode

acc_list = []
f1_list = []
missing_perc_list = []

files = os.listdir("../data/llm_prompt_outputs")

files_xls = [f for f in files if "falcon" in f]

for file in files_xls:
    df = pd.read_pickle("../data/llm_prompt_outputs/" + file)

    true_labels = []
    predicted_labels = []

    for index in range(df.shape[0]):
        true_labels_temp = df.loc[[index], ["true_label"]].values[0, 0]
        original_sent_temp = df.loc[[index], ["original_sent"]].values[0, 0]
        text_output_temp = df.loc[[index], ["text_output"]].values[0, 0]

        original_sent_split = original_sent_temp.split("\n")
        text_output_split = text_output_temp.split("\n")

        true_labels = true_labels + true_labels_temp

        sub_index_gold = 0
        sub_index_output = 0
        while sub_index_gold < len(
            original_sent_split
        ):  # or sub_index_output < len(text_output_split):
            try:
                text_output_token_label = text_output_split[sub_index_output].split(":")
                # print(original_sent_split[sub_index_gold], text_output_token_label[0], text_output_token_label[1], "AAA")
                if original_sent_split[sub_index_gold] == text_output_token_label[0]:
                    sub_index_output = sub_index_output + 1
                    predicted_labels.append(finer_ord_decode(text_output_token_label[1]))
                elif original_sent_split[sub_index_gold] in text_output_token_label[0]:
                    if text_output_token_label[0].endswith(
                        original_sent_split[sub_index_gold]
                    ):
                        sub_index_output = sub_index_output + 1
                    predicted_labels.append(finer_ord_decode(text_output_token_label[1]))
                else:
                    predicted_labels.append(-1)
            except:
                predicted_labels.append(-1)
            sub_index_gold = sub_index_gold + 1

    acc_list.append(accuracy_score(true_labels, predicted_labels))
    f1_list.append(f1_score(true_labels, predicted_labels, average="weighted"))
    missing_perc_list.append(
        (predicted_labels.count(-1) / len(predicted_labels)) * 100.0
    )
    print(classification_report(predicted_labels, true_labels))

print("f1 score mean: ", format(np.mean(f1_list), ".4f"))
print("f1 score std: ", format(np.std(f1_list), ".4f"))
print(
    "Percentage of cases when didn't follow instruction: ",
    format(np.mean(missing_perc_list), ".4f"),
    "\n",
)
