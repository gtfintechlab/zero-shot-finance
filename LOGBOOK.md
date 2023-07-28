# Research Logbook

## Friday, 2023-07-28

### Journal
- Talk to Agam on Monday about the way `instruct_pipeline.py` works ... I could be wrong but it seems like it is pushing through all 453 at once not batching/one-at-a-time? // Actually I think it is doing it one-at-a-time using the Pipeline since there is not a DataLoader provided.
- Could using `.batch_decode()` instead of `.decode()` in the instruction pipeline speed things up?

#### Vanilla Runtimes
Runtimes are found in the original research. There were 453 sentences in the test set for sentiment classification. Reported inference time was ~7s each so $453 \times 7 \times \frac{1}{60} =$ 52.85 minutes for completion -- which is similar to what I was getting!

### TODOs

* [ ] Use `get_financial_phrasebank()` to download the dataset from HuggingFace for the project
* [ ] See if `.batch_decode()` can speed up over `.decode` in `InstructionTextGenerationPipeline`
* [ ] Replace all the dashes with underscores e.g. FPB-sentiment-analysis-allagree with FPB_sentiment_analysis_allagree

## Thursday, 2023-07-27

### Journal
- turned off device mapping for inference and instead set to use one GPU for now
- dolly model inference is working with quantization and batching
- utils.py broken down into cuda_utils and model_utils

### TODOs
* [X] transformers pipelines can speed up inference?

```bash
/home/glenn/Data/miniconda3/envs/qllm/lib/python3.11/site-packages/transformers/pipelines/base.py:1089: UserWarning: You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset
  warnings.warn(

```
- convert the prompts to a dataset and use batch sizes
- tuning the batch sizes, how to determine batch size automagically?
- [Investigate continuous batching for transformer inference to speed up experiments](https://github.com/huggingface/text-generation-inference/tree/main/router)
- re-enable the device mapping?
- add functionality to create_train_test_data.py to download and extract the data zip file

# Useful Links

## Torch
- https://pytorch.org/docs/stable/notes/cuda.html

## Transformers
- https://huggingface.co/transformers/v3.0.2/model_doc/auto.html
- https://huggingface.co/docs/transformers/perf_infer_gpu_one
- https://huggingface.co/docs/transformers/main_classes/quantization

## Quantization
- https://github.com/TimDettmers/bitsandbytes
- https://github.com/huggingface/peft
- https://github.com/artidoro/qlora