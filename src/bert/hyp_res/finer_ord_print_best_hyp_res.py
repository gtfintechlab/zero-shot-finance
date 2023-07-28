import os

import pandas as pd

df = pd.read_csv(
    "../../../data/finer_ord/grid_search_results/roberta-large-best_1_0/results.csv"
)
df_temp = df.groupby(["Learning Rate", "Batch Size"], as_index=False).agg(
    {
        "Val F1": ["mean"],
        "Test F1": ["mean", "std"],
        "Fine Tuning Time(m)": ["mean"],
        "Test Labeling Time(m)": ["mean"],
    }
)
df_temp.columns = [
    "Learning Rate",
    "Batch Size",
    "mean Val F1 Score",
    "mean Test F1 Score",
    "std Test F1 Score",
    "mean Fine Tuning Time(m)",
    "mean Test Labeling Time(m)",
]

max_element = df_temp.iloc[df_temp["mean Val F1 Score"].idxmax()]
print(max_element)
# print(max_element['mean Test F1 Score'])
# print(format(max_element['std Test F1 Score'], '.4f'), "\n")
