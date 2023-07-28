"""
Falcon 7B Instruct: https://huggingface.co/tiiuae/falcon-7b-instruct
Falcon 40B Instruct: https://huggingface.co/tiiuae/falcon-40b-instruct
"""

import os
from datetime import date
from time import time

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

today = date.today()

# set gpu
os.environ["CUDA_VISIBLE_DEVICES"] = str("0")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device assigned: ", device)

model = "tiiuae/falcon-7b-instruct"  # "tiiuae/falcon-40b-instruct"

# get model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)


start_t = time()
# load training data
test_data_path = "../../data/finer_ord/test/test.csv"
data_df = pd.read_csv(test_data_path)

grouped_df = (
    data_df.groupby(["doc_idx", "sent_idx"])
    .agg({"gold_label": lambda x: list(x), "gold_token": lambda x: list(x)})
    .reset_index()
)
grouped_df.columns = ["doc_idx", "sent_idx", "gold_label", "gold_token"]


prompts_list = []

for index in range(grouped_df.shape[0]):
    token_list = grouped_df.loc[[index], ["gold_token"]].values[0, 0]
    label_list = grouped_df.loc[[index], ["gold_label"]].values[0, 0]
    sen = "\n".join(token_list)

    prompt = (
        "Discard all the previous instructions. Behave like you are an expert named entity identifier. Below a sentence is tokenized and each line contains a word token from the sentence. Identify 'Person', 'Location', and 'Organisation' from them and label them. If the entity is multi token use post-fix _B for the first label and _I for the remaining token labels for that particular entity. The start of the separate entity should always use _B post-fix for the label. If the token doesn't fit in any of those three categories or is not a named entity label it 'Other'. Do not combine words yourself. Use a colon to separate token and label. So the format should be token:label. \n\n"
        + sen
    )

    prompts_list.append(prompt)

# documentation: https://huggingface.co/docs/transformers/v4.29.1/en/main_classes/text_generation
res = pipeline(
    prompts_list,
    max_new_tokens=512,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)

output_list = []

for i in range(len(res)):
    token_list = grouped_df.loc[[i], ["gold_token"]].values[0, 0]
    label_list = grouped_df.loc[[i], ["gold_label"]].values[0, 0]
    sen = "\n".join(token_list)
    output_list.append(
        [label_list, sen, res[i][0]["generated_text"][len(prompts_list[i]) :]]
    )


results = pd.DataFrame(
    output_list, columns=["true_label", "original_sent", "text_output"]
)
time_taken = int((time() - start_t) / 60.0)

results.to_pickle(
    f'../data/llm_prompt_outputs/falcon_7b_{today.strftime("%d_%m_%Y")}_{time_taken}'
)
