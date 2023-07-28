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
seeds = [5768, 78516, 944601]

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


for seed in [5768, 78516, 944601]:
    # assign seed to numpy and PyTorch
    torch.manual_seed(seed)
    np.random.seed(seed)

    start_t = time()
    # load test data
    data_category = "FPB-sentiment-analysis-allagree"
    test_data_path = (
        "../data/test/" + data_category + "-test" + "-" + str(seed) + ".xlsx"
    )
    data_df = pd.read_excel(test_data_path)

    sentences = data_df["sentence"].to_list()
    labels = data_df["label"].to_numpy()

    prompts_list = []
    for sen in sentences:
        prompt = (
            "Discard all the previous instructions. Behave like you are an expert sentence sentiment classifier. Classify the following sentence into 'NEGATIVE', 'POSITIVE', or 'NEUTRAL' class. Label 'NEGATIVE' if it is corresponding to negative sentiment, 'POSITIVE' if it is corresponding to positive sentiment, or 'NEUTRAL' if the sentiment is neutral. Provide the label in the first line and provide a short explanation in the second line. The sentence: "
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
        # print(res[i][0]['generated_text'][len(prompts_list[i]):])
        output_list.append(
            [
                labels[i],
                sentences[i],
                res[i][0]["generated_text"][len(prompts_list[i]) :],
            ]
        )

    results = pd.DataFrame(
        output_list, columns=["true_label", "original_sent", "text_output"]
    )
    time_taken = int((time() - start_t) / 60.0)

    results.to_csv(
        f'../data/llm_prompt_outputs/falcon_7b_{seed}_{today.strftime("%d_%m_%Y")}_{time_taken}.csv',
        index=False,
    )
