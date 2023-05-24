import os
import torch
from h2oai_pipeline import H2OTextGenerationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

import pandas as pd
import numpy as np
from time import time
from datetime import date


today = date.today()
seeds = [5768, 78516, 944601]

# set gpu
os.environ["CUDA_VISIBLE_DEVICES"] = str("0")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Device assigned: ", device)

# get model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("h2oai/h2ogpt-oasst1-512-12b", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("h2oai/h2ogpt-oasst1-512-12b", torch_dtype=torch.bfloat16, device_map="auto")

# get pipeline ready for instruction text generation
generate_text = H2OTextGenerationPipeline(model=model, tokenizer=tokenizer, max_length=512)




start_t = time()
# load training data
test_data_path = "../data/test/test.csv"
data_df = pd.read_csv(test_data_path)

grouped_df = data_df.groupby(['doc_idx', 'sent_idx']).agg({'gold_label':lambda x: list(x), 'gold_token':lambda x: list(x)}).reset_index()
grouped_df.columns = ['doc_idx', 'sent_idx', 'gold_label', 'gold_token']



prompts_list = []

for index in range(grouped_df.shape[0]):
    token_list = grouped_df.loc[[index],['gold_token']].values[0, 0]
    label_list = grouped_df.loc[[index],['gold_label']].values[0, 0]
    sen = '\n'.join(token_list)

    prompt = "Discard all the previous instructions. Behave like you are an expert named entity identifier. Below a sentence is tokenized and each line contains a word token from the sentence. Identify 'Person', 'Location', and 'Organisation' from them and label them. If the entity is multi token use post-fix _B for the first label and _I for the remaining token labels for that particular entity. The start of the separate entity should always use _B post-fix for the label. If the token doesn't fit in any of those three categories or is not a named entity label it 'Other'. Do not combine words yourself. Use a colon to separate token and label. So the format should be token:label. \n\n" + sen
    
    prompts_list.append(prompt)

res = generate_text(prompts_list)

output_list = []

for i in range(len(res)):
    token_list = grouped_df.loc[[i],['gold_token']].values[0, 0]
    label_list = grouped_df.loc[[i],['gold_label']].values[0, 0]
    sen = '\n'.join(token_list)
    output_list.append([label_list, sen, res[i][0]['generated_text']])


results = pd.DataFrame(output_list, columns=["true_label", "original_sent", "text_output"])
time_taken = int((time() - start_t)/60.0)

results.to_pickle(f'../data/llm_prompt_outputs/h2o_{today.strftime("%d_%m_%Y")}_{time_taken}')
