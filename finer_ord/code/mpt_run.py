"""
Add the model once obvious errors are fixed in next few weeks
# https://huggingface.co/mosaicml/mpt-7b-instruct
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline, AutoConfig

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
tokenizer = AutoTokenizer.from_pretrained('mosaicml/mpt-7b-instruct', padding_side="left")
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

model = AutoModelForCausalLM.from_pretrained(
  'mosaicml/mpt-7b-instruct',
  trust_remote_code=True#, torch_dtype=torch.bfloat16
)
model.to(device='cuda:0')

# get pipeline ready for instruction text generation
generate_text = TextGenerationPipeline(model=model, tokenizer=tokenizer, max_length=512)

print(generate_text(['What is a quoll?']))

exit(0)

for seed in [5768, 78516, 944601]: 

    # assign seed to numpy and PyTorch
    torch.manual_seed(seed)
    np.random.seed(seed) 

    start_t = time()
    # load test data
    test_data_path = "../data/test/lab-manual-split-combine-test" + "-" + str(seed) + ".xlsx"
    data_df = pd.read_excel(test_data_path)


    sentences = data_df['sentence'].to_list()
    labels = data_df['label'].to_numpy()

    prompts_list = []
    for sen in sentences:
        prompt = "Discard all the previous instructions. Behave like you are an expert sentence classifier. Classify the following sentence from FOMC into 'HAWKISH', 'DOVISH', or 'NEUTRAL' class. Label 'HAWKISH' if it is corresponding to tightening of the monetary policy, 'DOVISH' if it is corresponding to easing of the monetary policy, or 'NEUTRAL' if the stance is neutral. Provide the label in the first line and provide a short explanation in the second line. The sentence: " + sen
        prompts_list.append(prompt)

    res = generate_text(prompts_list, pad_token_id=tokenizer.eos_token_id)

    output_list = []

    for i in range(len(res)):
        output_list.append([labels[i], sentences[i], res[i][0]['generated_text']])


    results = pd.DataFrame(output_list, columns=["true_label", "original_sent", "text_output"])
    time_taken = int((time() - start_t)/60.0)
    results.to_csv(f'../data/llm_prompt_outputs/mpt7b_{seed}_{today.strftime("%d_%m_%Y")}_{time_taken}.csv', index=False)
