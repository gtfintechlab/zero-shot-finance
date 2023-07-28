import os
import sys
from time import sleep, time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForSequenceClassification,
    BertTokenizerFast,
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
    XLNetForSequenceClassification,
    XLNetTokenizerFast,
)

# sys.path.append('..')


def fine_tune_plm(
    gpu_numbers: str,
    train_data_path: str,
    test_data_path: str,
    language_model_to_use: str,
    seed: int,
    batch_size: int,
    learning_rate: float,
    save_model_path: str,
):
    """
    Description: Run experiment over particular batch size, learning rate and seed
    """
    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_numbers)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device assigned: ", device)

    # load training data
    data_df = pd.read_excel(train_data_path)
    sentences = data_df["sentence"].to_list()
    labels = data_df["label"].to_numpy()

    # load test data
    data_df_test = pd.read_excel(test_data_path)
    sentences_test = data_df_test["sentence"].to_list()
    labels_test = data_df_test["label"].to_numpy()

    # load tokenizer
    try:
        if language_model_to_use == "roberta":
            tokenizer = RobertaTokenizerFast.from_pretrained(
                "roberta-base", do_lower_case=True, do_basic_tokenize=True
            )
        elif language_model_to_use == "roberta-large":
            tokenizer = RobertaTokenizerFast.from_pretrained(
                "roberta-large", do_lower_case=True, do_basic_tokenize=True
            )
        else:
            return -1
    except Exception as e:
        print(e)
        sleep(600)
        if language_model_to_use == "roberta":
            tokenizer = RobertaTokenizerFast.from_pretrained(
                "roberta-base", do_lower_case=True, do_basic_tokenize=True
            )
        elif language_model_to_use == "roberta-large":
            tokenizer = RobertaTokenizerFast.from_pretrained(
                "roberta-large", do_lower_case=True, do_basic_tokenize=True
            )
        else:
            return -1

    max_length = 0
    sentence_input = []
    labels_output = []
    for i, sentence in enumerate(sentences):
        if isinstance(sentence, str):
            tokens = tokenizer(sentence)["input_ids"]
            sentence_input.append(sentence)
            max_length = max(max_length, len(tokens))
            labels_output.append(labels[i])
        else:
            pass
    max_length = 256
    if language_model_to_use == "flangroberta":
        max_length = 128
    tokens = tokenizer(
        sentence_input,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    labels = np.array(labels_output)

    input_ids = tokens["input_ids"]
    attention_masks = tokens["attention_mask"]
    labels = torch.LongTensor(labels)
    dataset = TensorDataset(input_ids, attention_masks, labels)
    val_length = int(len(dataset) * 0.2)
    train_length = len(dataset) - val_length
    print(f"Train Size: {train_length}, Validation Size: {val_length}")
    experiment_results = []

    # assign seed to numpy and PyTorch
    torch.manual_seed(seed)
    np.random.seed(seed)

    # select language model
    try:
        if language_model_to_use == "roberta":
            model = RobertaForSequenceClassification.from_pretrained(
                "roberta-base", num_labels=3
            ).to(device)
        elif language_model_to_use == "roberta-large":
            model = RobertaForSequenceClassification.from_pretrained(
                "roberta-large", num_labels=3
            ).to(device)
        else:
            return -1
    except:
        sleep(600)
        if language_model_to_use == "roberta":
            model = RobertaForSequenceClassification.from_pretrained(
                "roberta-base", num_labels=3
            ).to(device)
        elif language_model_to_use == "roberta-large":
            model = RobertaForSequenceClassification.from_pretrained(
                "roberta-large", num_labels=3
            ).to(device)
        else:
            return -1

    # create train-val split
    train, val = torch.utils.data.random_split(
        dataset=dataset, lengths=[train_length, val_length]
    )
    dataloaders_dict = {
        "train": DataLoader(train, batch_size=batch_size, shuffle=True),
        "val": DataLoader(val, batch_size=batch_size, shuffle=True),
    }
    print(train_length, val_length)
    # select optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    max_num_epochs = 100
    max_early_stopping = 7
    early_stopping_count = 0
    best_ce = float("inf")
    best_accuracy = float("-inf")
    best_f1 = float("-inf")

    eps = 1e-2

    start_fine_tuning = time()

    for epoch in range(max_num_epochs):
        if early_stopping_count >= max_early_stopping:
            break
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                early_stopping_count += 1
            else:
                model.eval()

            curr_ce = 0
            curr_accuracy = 0
            actual = torch.tensor([]).long().to(device)
            pred = torch.tensor([]).long().to(device)

            for input_ids, attention_masks, labels in dataloaders_dict[phase]:
                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_masks,
                        labels=labels,
                    )
                    loss = outputs.loss
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    else:
                        curr_ce += loss.item() * input_ids.size(0)
                        curr_accuracy += torch.sum(
                            torch.max(outputs.logits, 1)[1] == labels
                        ).item()
                        actual = torch.cat([actual, labels], dim=0)
                        pred = torch.cat([pred, torch.max(outputs.logits, 1)[1]], dim=0)
            if phase == "val":
                curr_ce = curr_ce / len(val)
                curr_accuracy = curr_accuracy / len(val)
                currF1 = f1_score(
                    actual.cpu().detach().numpy(),
                    pred.cpu().detach().numpy(),
                    average="weighted",
                )
                if currF1 >= best_f1 + eps:
                    best_f1 = currF1
                    best_accuracy = curr_accuracy
                    best_ce = curr_ce
                    early_stopping_count = 0
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                        },
                        "best_model.pt",
                    )
                print("Val CE: ", curr_ce)
                print("Val Accuracy: ", curr_accuracy)
                print("Val F1: ", currF1)
                print("Early Stopping Count: ", early_stopping_count)
    training_time_taken = (time() - start_fine_tuning) / 60.0
    ## ------------------testing---------------------
    checkpoint = torch.load("best_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])

    start_test_labeling = time()

    sentence_input_test = []
    labels_output_test = []
    for i, sentence in enumerate(sentences_test):
        if isinstance(sentence, str):
            tokens = tokenizer(sentence)["input_ids"]
            sentence_input_test.append(sentence)
            labels_output_test.append(labels_test[i])
        else:
            pass

    tokens_test = tokenizer(
        sentence_input_test,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    labels_test = np.array(labels_output_test)

    input_ids_test = tokens_test["input_ids"]
    attention_masks_test = tokens_test["attention_mask"]
    labels_test = torch.LongTensor(labels_test)
    dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)

    dataloaders_dict_test = {
        "test": DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    }
    test_ce = 0
    test_accuracy = 0
    actual = torch.tensor([]).long().to(device)
    pred = torch.tensor([]).long().to(device)
    for input_ids, attention_masks, labels in dataloaders_dict_test["test"]:
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids, attention_mask=attention_masks, labels=labels
            )
            loss = outputs.loss
            test_ce += loss.item() * input_ids.size(0)
            test_accuracy += torch.sum(torch.max(outputs.logits, 1)[1] == labels).item()
            actual = torch.cat([actual, labels], dim=0)
            pred = torch.cat([pred, torch.max(outputs.logits, 1)[1]], dim=0)
    test_time_taken = (time() - start_test_labeling) / 60.0
    test_ce = test_ce / len(dataset_test)
    test_accuracy = test_accuracy / len(dataset_test)
    test_f1 = f1_score(
        actual.cpu().detach().numpy(), pred.cpu().detach().numpy(), average="weighted"
    )
    experiment_results = [
        seed,
        learning_rate,
        batch_size,
        best_ce,
        best_accuracy,
        best_f1,
        test_ce,
        test_accuracy,
        test_f1,
        training_time_taken,
        test_time_taken,
    ]

    # save model
    if save_model_path != None:
        model.save_pretrained(save_model_path)
        tokenizer.save_pretrained(save_model_path)

    return experiment_results


def train_lm_experiments(
    gpu_numbers: str,
    train_data_path_prefix: str,
    test_data_path_prefix: str,
    language_model_to_use: str,
    data_category: str,
):
    """
    Description: Run experiments over different batch sizes, learning rates and seeds to find best hyperparameters
    """
    results = []
    seeds = [5768, 78516, 944601]
    batch_sizes = [32, 16, 8, 4]
    learning_rates = [1e-4, 1e-5, 1e-6, 1e-7]
    count = 0
    for i, seed in enumerate(seeds):
        for k, batch_size in enumerate(batch_sizes):
            for j, learning_rate in enumerate(learning_rates):
                count += 1
                print(
                    f"Experiment {count} of {len(seeds) * len(batch_sizes) * len(learning_rates)}:"
                )

                train_data_path = train_data_path_prefix + "-" + str(seed) + ".xlsx"
                test_data_path = test_data_path_prefix + "-" + str(seed) + ".xlsx"

                results.append(
                    fine_tune_plm(
                        gpu_numbers,
                        train_data_path,
                        test_data_path,
                        language_model_to_use,
                        seed,
                        batch_size,
                        learning_rate,
                        None,
                    )
                )
                df = pd.DataFrame(
                    results,
                    columns=[
                        "Seed",
                        "Learning Rate",
                        "Batch Size",
                        "Val Cross Entropy",
                        "Val Accuracy",
                        "Val F1 Score",
                        "Test Cross Entropy",
                        "Test Accuracy",
                        "Test F1 Score",
                        "Fine Tuning Time(m)",
                        "Test Labeling Time(m)",
                    ],
                )
                df.to_excel(
                    f"../data/grid_search_results/final_{data_category}_{language_model_to_use}.xlsx",
                    index=False,
                )


if __name__ == "__main__":
    start_t = time()

    # experiments
    for language_model_to_use in ["roberta", "roberta-large"]:  # provide list of models
        for data_category in [
            "FPB-sentiment-analysis-allagree"
        ]:  # provide list of multiple data categories if it has
            train_data_path_prefix = "../data/train/" + data_category + "-train"
            test_data_path_prefix = "../data/test/" + data_category + "-test"
            train_lm_experiments(
                gpu_numbers="0",
                train_data_path_prefix=train_data_path_prefix,
                test_data_path_prefix=test_data_path_prefix,
                language_model_to_use=language_model_to_use,
                data_category=data_category,
            )

    print((time() - start_t) / 60.0)
