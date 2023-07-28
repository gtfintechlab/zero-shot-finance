instruction_sentiment_analysis = "Discard all the previous instructions. Behave like you are an expert sentence sentiment classifier. Classify the following sentence into 'NEGATIVE', 'POSITIVE', or 'NEUTRAL' class. Label 'NEGATIVE' if it is corresponding to negative sentiment, 'POSITIVE' if it is corresponding to positive sentiment, or 'NEUTRAL' if the sentiment is neutral. Provide the label in the first line and provide a short explanation in the second line. The sentence: "
instruction_numclaim_detection = "Discard all the previous instructions. Behave like you are an expert sentence sentiment classifier. Classify the following sentence into 'INCLAIM', or 'OUTOFCLAIM' class. Label 'INCLAIM' if consist of a claim and not just factual past or present information, or 'OUTOFCLAIM' if it has just factual past or present information. Provide the label in the first line and provide a short explanation in the second line. The sentence: "
instruction_fomc_communication = "Discard all the previous instructions. Behave like you are an expert sentence classifier. Classify the following sentence from FOMC into 'HAWKISH', 'DOVISH', or 'NEUTRAL' class. Label 'HAWKISH' if it is corresponding to tightening of the monetary policy, 'DOVISH' if it is corresponding to easing of the monetary policy, or 'NEUTRAL' if the stance is neutral. Provide the label in the first line and provide a short explanation in the second line. The sentence: "
instruction_finer_ord = "Discard all the previous instructions. Behave like you are an expert named entity identifier. Below a sentence is tokenized and each line contains a word token from the sentence. Identify 'Person', 'Location', and 'Organisation' from them and label them. If the entity is multi token use post-fix _B for the first label and _I for the remaining token labels for that particular entity. The start of the separate entity should always use _B post-fix for the label. If the token doesn't fit in any of those three categories or is not a named entity label it 'Other'. Do not combine words yourself. Use a colon to separate token and label. So the format should be token:label. \n\n"

task_data_map = {
    "sentiment_analysis": {
        "data_category": "FPB-sentiment-analysis-allagree",
        "instruction": instruction_sentiment_analysis,
    },
    "numclaim": {
        "data_category": "FPB-sentiment-analysis-allagree",
        "instruction": instruction_numclaim_detection,
    },
    "fomc_communication": {
        "data_category": "lab-manual-split-combine-test",
        "instruction": instruction_fomc_communication,
    },
    "finer_ord": {"data_category": "test.csv", "instruction": instruction_finer_ord},
}
