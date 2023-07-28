import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from utils.results.decode import fomc_communication_decode

acc_list = []
f1_list = []
missing_perc_list = []

files = os.listdir("../data/llm_prompt_outputs")

files_xls = [f for f in files if "palm" in f]

for file in files_xls:
    df = pd.read_csv("../data/llm_prompt_outputs/" + file)

    df["predicted_label"] = df["text_output"].apply(lambda x: fomc_communication_decode(x))
    acc_list.append(accuracy_score(df["true_label"], df["predicted_label"]))
    f1_list.append(
        f1_score(df["true_label"], df["predicted_label"], average="weighted")
    )
    missing_perc_list.append(
        (len(df[df["predicted_label"] == -1]) / df.shape[0]) * 100.0
    )

print("f1 score mean: ", format(np.mean(f1_list), ".4f"))
print("f1 score std: ", format(np.std(f1_list), ".4f"))
print(
    "Percentage of cases when didn't follow instruction: ",
    format(np.mean(missing_perc_list), ".4f"),
    "\n",
)
