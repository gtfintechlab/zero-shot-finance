# Research Logbook

## Thursday, 2023-07-27

### Journal
- turned off device mapping for inference and instead set to use one GPU for now
- dolly model inference is working with quantization and batching
- utils.py broken down into cuda_utils and model_utils

### TODOs
- tuning the batch sizes, how to determine batch size automagically?
- [Investigate continuous batching for transformer inference to speed up experiments](https://github.com/huggingface/text-generation-inference/tree/main/router)
- re-enable the device mapping?

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