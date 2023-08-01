import os
import sys
from datetime import date
from time import sleep, time

import openai
import pandas as pd

sys.path.insert(0, "/home/research/git repos/zero-shot-finance")
from api_keys import APIKeyConstants

today = date.today()
openai.api_key = APIKeyConstants.OPENAI_API_KEY

for seed in [5768, 78516, 944601]:
    for data_category in ["numclaim"]:
        start_t = time()
        # load training data
        test_data_path = (
            "../data/test/" + data_category + "-test" + "-" + str(seed) + ".xlsx"
        )
        data_df = pd.read_excel(test_data_path)

        sentences = data_df["text"].to_list()
        labels = data_df["label"].to_numpy()

        output_list = []
        for i in range(len(sentences)):
            sen = sentences[i]
            message = (
                "Discard all the previous instructions. Behave like you are an expert sentence sentiment classifier. Classify the following sentence into 'INCLAIM', or 'OUTOFCLAIM' class. Label 'INCLAIM' if consist of a claim and not just factual past or present information, or 'OUTOFCLAIM' if it has just factual past or present information. Provide the label in the first line and provide a short explanation in the second line. The sentence: "
                + sen
            )

            prompt_json = [
                {"role": "user", "content": message},
            ]
            try:
                chat_completion = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=prompt_json,
                    temperature=0.0,
                    max_tokens=1000,
                )
            except Exception as e:
                print(e)
                i = i - 1
                sleep(10.0)

            answer = chat_completion.choices[0].message.content

            output_list.append([labels[i], sen, answer])
            sleep(6.0)

        results = pd.DataFrame(
            output_list, columns=["true_label", "original_sent", "text_output"]
        )

        time_taken = int((time() - start_t) / 60.0)
        results.to_csv(
            f'../data/llm_prompt_outputs/gpt4_{data_category}_{seed}_{today.strftime("%d_%m_%Y")}_{time_taken}.csv',
            index=False,
        )
