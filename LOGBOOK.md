# Research Logbook

## TODOs
* [ ] Use `get_financial_phrasebank()` to download the dataset from HuggingFace for the project
* [ ] See if `.batch_decode()` can speed up over `.decode` in `InstructionTextGenerationPipeline`
* [ ] Replace all the dashes with underscores e.g. FPB-sentiment-analysis-allagree with FPB_sentiment_analysis_allagree
* [ ] Convert data used for "finer_ord" from `.csv.` to `.xlsx` so I can use `pd.read_xlsx` instead
* [ ] For dolly, give a control flow to separate the way "finer_ord" is handled compared to the other tasks
* [ ] [Investigate continuous batching for transformer inference to speed up experiments](https://github.com/huggingface/text-generation-inference/tree/main/router)
* [X] add functionality to create_train_test_data.py to download and extract the data zip file

## Askers?
* [ ] Discuss how `instruct_pipeline.py` works ... I could be wrong but it seems like it is pushing through all 453 at once not batching/one-at-a-time? // Actually I think it is doing it one-at-a-time using the Pipeline since there is not a DataLoader provided.
* [ ] Is the numclaim data is just the same as the sentiment analysis data ... there is no data folder for numclaims
* [ ] What is the deal with `lab-manual-split-combine-test` for FOMC data, 'test.csv' for finder_ord

## Useful Links

### Torch
- https://pytorch.org/docs/stable/notes/cuda.html

### Transformers
- https://huggingface.co/transformers/v3.0.2/model_doc/auto.html
- https://huggingface.co/docs/transformers/perf_infer_gpu_one
- https://huggingface.co/docs/transformers/main_classes/quantization

### Quantization
- https://github.com/TimDettmers/bitsandbytes
- https://github.com/huggingface/peft
- https://github.com/artidoro/qlora

## Journal

### Friday, 2023-07-28
- Could using `.batch_decode()` instead of `.decode()` in the instruction pipeline speed things up?
- Runtimes are found in the original research. There were 453 sentences in the test set for sentiment classification. Reported inference time was ~7s each so $453 \times 7 \times \frac{1}{60} =$ 52.85 minutes for completion -- which is similar to what I was getting!

## Thursday, 2023-07-27
- turned off device mapping for inference and instead set to use one GPU for now
- dolly model inference is working with quantization and batching
- utils.py broken down into cuda_utils and model_utils