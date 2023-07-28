import json
import logging
import os
import shutil
import sys
from time import time
from typing import Dict, Union

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import RobertaForTokenClassification, RobertaTokenizerFast, pipeline

logging.basicConfig()


class BERTForNer:
    def __init__(self, config_path):
        config = json.load(open(config_path))

        self.results = []
        self.seeds = config["seeds"]
        self.epsilon = config["epsilon"]
        self.batch_sizes = config["batch_sizes"]
        self.learning_rates = config["learning_rates"]

        self.num_epochs = config["num_epochs"]
        self.early_stopping_limit = config["early_stopping_limit"]

        self.train_path = config["train_data_path"]
        self.val_path = config["val_data_path"]
        self.test_path = config["test_data_path"]
        self.results_save_path = config["results_save_path"]
        self.experiment_name = config["experiment_name"]
        self.experiment_version = config["experiment_version"]
        self.language_model = config["language-model"]

        gpu = config["gpu"]
        self.device = (
            torch.device(f"cuda:{gpu}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            self.language_model,
            do_lower_case=False,
            do_basic_tokenize=True,
            add_prefix_space=True,
        )
        # self.tokenizer.save_pretrained("./final_model")

        self.criterion = None
        self.int2str = config["int2str"]
        self.int2str = {int(k): v for k, v in self.int2str.items()}

        self.current_experiment_state: Dict[str, Union[int, float, None]] = {
            "seed": None,
            "learning_rate": None,
            "batch_size": None,
        }

        self.best_val_accuracy = float("-inf")
        self.best_val_f1 = float("-inf")
        self.best_val_ce = float("inf")

        self.fine_tuning_time = float("inf")

        self.label_all_tokens = False
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        dir_name = f"{self.experiment_name}_{self.experiment_version}".replace(".", "_")
        dir_path = os.path.join(self.results_save_path, dir_name)

        self.dir_path = dir_path

        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.mkdir(dir_path)

    def load_data(self, file_path: str, int2str: Dict[int, str]):
        """

        Args:
            file_path: Data file path
            int2str: Mapping of int label to str label

        Returns:

        Description: Data must contain these three columns.
            (1) sentence_id: Index of the sentence
            (2) token: token of sentence
            (3) label: NER label for token

        """
        df = pd.read_csv(file_path)
        df.label = df.label.map(int2str)
        df.dropna(inplace=True)

        df_sentences = df.groupby("uuid").agg({"token": list, "label": list})
        sentences = df_sentences.token.tolist()
        sentences_tags = df_sentences.label.tolist()

        max_length = 0
        dropped_sentences = 0
        filtered_sentences = []
        filtered_sentences_tags = []
        for i, sentence in enumerate(sentences):
            try:
                tokens = self.tokenizer(sentence, is_split_into_words=True)
                sent_len = len(tokens["input_ids"])
                if sent_len <= 512:
                    filtered_sentences.append(sentence)
                    filtered_sentences_tags.append(sentences_tags[i])
                    max_length = max(max_length, sent_len)
                else:
                    dropped_sentences += 1
            except Exception as e:
                self.logger.error(
                    f"Failed for sentence: {sentence} with exception: {e}"
                )

        self.logger.info(
            f"Dropped {dropped_sentences} because of length greater than 512"
        )
        sentences = filtered_sentences
        sentences_tags = filtered_sentences_tags

        label_list = list(df.label.unique())
        self.logger.info(f"Label list is: {label_list}")
        label_list.sort()
        str_to_int = {l: i for i, l in enumerate(label_list)}
        int_to_str = {i: l for (l, i) in str_to_int.items()}
        tokenized_inputs = self.tokenizer(
            sentences,
            max_length=max_length,
            padding="max_length",
            is_split_into_words=True,
            return_tensors="pt",
        )
        input_ids = tokenized_inputs["input_ids"]
        attention_masks = tokenized_inputs["attention_mask"]
        labels = []
        temp_global_list_labels = set()
        for i, label in enumerate(sentences_tags):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(str_to_int[label[word_idx]])
                else:
                    label_ids.append(
                        str_to_int[label[word_idx]] if self.label_all_tokens else -100
                    )
                previous_word_idx = word_idx
            labels.append(label_ids)
        labels = torch.LongTensor(labels)
        return TensorDataset(input_ids, attention_masks, labels), str_to_int, int_to_str

    def set_current_lr(self, lr: float):
        self.current_experiment_state["learning_rate"] = lr

    def set_current_seed(self, seed: int):
        self.current_experiment_state["seed"] = seed

    def set_current_batch_size(self, seed: int):
        self.current_experiment_state["batch_size"] = seed

    def set_criterion(self, train_str2int: Dict[str, int]):
        num_labels: int = len(train_str2int)
        weights = torch.ones(num_labels).to(self.device) * 1 / (num_labels - 1)
        weights[train_str2int["O"]] = 0.001
        weights[train_str2int["PER_B"]] = 0.1353 - 0.001 / 6
        weights[train_str2int["PER_I"]] = 0.0911 - 0.001 / 6
        weights[train_str2int["LOC_B"]] = 0.1592 - 0.001 / 6
        weights[train_str2int["LOC_I"]] = 0.0476 - 0.001 / 6
        weights[train_str2int["ORG_B"]] = 0.3338 - 0.001 / 6
        weights[train_str2int["ORG_I"]] = 0.2330 - 0.001 / 6

        self.criterion = torch.nn.CrossEntropyLoss(weight=weights)

        self.logger.info(f"Classes: {train_str2int}, Class weights: {weights}")

        return self.criterion

    def get_current_lr(self) -> float:
        assert (
            self.current_experiment_state["learning_rate"] is not None
        ), f"Learning rate not set for the experiment yet"
        return self.current_experiment_state["learning_rate"]

    def get_current_batch_size(self) -> float:
        assert (
            self.current_experiment_state["batch_size"] is not None
        ), f"Batch size not set for the experiment yet"
        return self.current_experiment_state["batch_size"]

    def get_current_seed(self) -> int:
        assert (
            self.current_experiment_state["seed"] is not None
        ), f"Seed not set for the experiment yet"
        return self.current_experiment_state["seed"]

    def get_criterion(self):
        assert self.criterion is not None, f"Criterion not set yet"
        return self.criterion

    def fine_tune(
        self, model, optimizer, dataloaders_dict, train_str2int: Dict[str, int]
    ):
        seed = self.get_current_seed()
        num_labels: int = len(train_str2int)
        criterion = self.get_criterion()

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.best_val_f1 = 0.0
        early_stopping_count = 0

        start_fine_tuning = time()
        for _ in tqdm(range(self.num_epochs), desc="# Epochs"):
            if early_stopping_count >= self.early_stopping_limit:
                break
            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()
                    early_stopping_count += 1
                else:
                    model.eval()
                curr_total = 0
                curr_correct = 0
                curr_ce = 0
                actual = np.array([])
                pred = np.array([])
                for input_ids, attention_masks, labels in tqdm(dataloaders_dict[phase]):
                    input_ids = input_ids.to(self.device)
                    attention_masks = attention_masks.to(self.device)
                    labels = labels.to(self.device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_masks,
                            labels=labels,
                        )
                        active_loss = attention_masks.view(-1) == 1
                        logits = outputs.logits
                        active_logits = logits.view(-1, num_labels)
                        active_labels = torch.where(
                            active_loss,
                            labels.view(-1),
                            torch.tensor(criterion.ignore_index).type_as(labels),
                        )
                        loss = criterion(active_logits, active_labels)

                        if phase == "train":
                            loss.backward()
                            optimizer.step()
                        else:
                            curr_pred = (
                                outputs.logits.argmax(dim=-1)
                                .detach()
                                .cpu()
                                .clone()
                                .numpy()
                            )
                            curr_actual = labels.detach().cpu().clone().numpy()
                            true_predictions = np.concatenate(
                                [
                                    [
                                        p
                                        for (p, l) in zip(
                                            sentence_preds, sentence_labels
                                        )
                                        if l != -100
                                    ]
                                    for sentence_preds, sentence_labels in zip(
                                        curr_pred, curr_actual
                                    )
                                ]
                            )
                            true_labels = np.concatenate(
                                [
                                    [
                                        l
                                        for (p, l) in zip(
                                            sentence_preds, sentence_labels
                                        )
                                        if l != -100
                                    ]
                                    for sentence_preds, sentence_labels in zip(
                                        curr_pred, curr_actual
                                    )
                                ]
                            )
                            curr_correct += np.sum(true_predictions == true_labels)
                            curr_total += len(true_predictions)
                            curr_ce += loss.item() * input_ids.size(0)
                            actual = np.concatenate([actual, true_labels], axis=0)
                            pred = np.concatenate([pred, true_predictions], axis=0)
                if phase == "val":
                    curr_accuracy = curr_correct / curr_total
                    curr_f1 = f1_score(actual, pred, average="weighted")
                    curr_ce = curr_ce / len(dataloaders_dict[phase])
                    if curr_f1 >= self.best_val_f1 + self.epsilon:
                        self.best_val_f1 = curr_f1
                        self.best_val_ce = curr_ce
                        self.best_val_accuracy = curr_accuracy
                        early_stopping_count = 0
                        torch.save(
                            {
                                "model_state_dict": model.state_dict(),
                            },
                            "best_model.pt",
                        )
                        # model.save_pretrained("./final_model")
                    self.logger.info(f"Val Cross Entropy: {curr_ce}")
                    self.logger.info(f"Val Accuracy: {curr_accuracy}")
                    self.logger.info(f"Val F1: {curr_f1}")
                    self.logger.info(f"Early Stopping Count: {early_stopping_count}")
        self.fine_tuning_time = (time() - start_fine_tuning) / 60.0
        # classifier = pipeline("ner", model=model, tokenizer=self.tokenizer, device=0, framework="pt")

        # example = "My name is Sarah and I live in London"
        # ner_results = classifier(example)
        # for token in ner_results:
        #     print(token["word"], token["entity"])

    def test(self, model, optimizer, dataloaders_dict, train_str2int: Dict[str, int]):
        model = RobertaForTokenClassification.from_pretrained(
            self.language_model, num_labels=7
        ).to(self.device)
        checkpoint = torch.load("best_model.pt")
        model.load_state_dict(checkpoint["model_state_dict"])

        lr = self.get_current_lr()
        seed = self.get_current_seed()
        bs = self.get_current_batch_size()

        criterion = self.get_criterion()

        start_test_labeling = time()

        num_labels = len(train_str2int)
        test_total = 0
        test_correct = 0
        test_ce = 0
        actual = np.array([])
        pred = np.array([])
        for input_ids, attention_masks, labels in dataloaders_dict["test"]:
            input_ids = input_ids.to(self.device)
            attention_masks = attention_masks.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad()
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_masks)
                active_loss = attention_masks.view(-1) == 1
                logits = outputs.logits
                active_logits = logits.view(-1, num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(criterion.ignore_index).type_as(labels),
                )
                loss = criterion(active_logits, active_labels)
                curr_pred = outputs.logits.argmax(dim=-1).detach().cpu().clone().numpy()
                curr_actual = labels.detach().cpu().clone().numpy()
                true_predictions = np.concatenate(
                    [
                        [
                            p
                            for (p, l) in zip(sentence_preds, sentence_labels)
                            if l != -100
                        ]
                        for sentence_preds, sentence_labels in zip(
                            curr_pred, curr_actual
                        )
                    ]
                )
                true_labels = np.concatenate(
                    [
                        [
                            l
                            for (p, l) in zip(sentence_preds, sentence_labels)
                            if l != -100
                        ]
                        for sentence_preds, sentence_labels in zip(
                            curr_pred, curr_actual
                        )
                    ]
                )
                test_total += len(true_predictions)
                test_ce += loss.item() * input_ids.size(0)
                test_correct += np.sum(true_predictions == true_labels)
                actual = np.concatenate([actual, true_labels], axis=0)
                pred = np.concatenate([pred, true_predictions], axis=0)

        test_time_taken = (time() - start_test_labeling) / 60.0
        test_accuracy = test_correct / test_total
        test_ce = test_ce / len(dataloaders_dict["test"])
        test_f1 = f1_score(actual, pred, average="weighted")
        confusion_matrix_temp = confusion_matrix(
            actual, pred, labels=list(train_str2int.values())
        )
        print(confusion_matrix_temp)

        report = classification_report(
            actual,
            pred,
            labels=list(train_str2int.values()),
            target_names=list(train_str2int.keys()),
            digits=4,
            zero_division=0,
        )
        report_json = classification_report(
            actual,
            pred,
            labels=list(train_str2int.values()),
            target_names=list(train_str2int.keys()),
            digits=4,
            output_dict=True,
            zero_division=0,
        )

        report_filename = f"report_seed_{self.get_current_seed()}.csv"
        report_filepath = os.path.join(self.dir_path, report_filename)
        pd.DataFrame(report_json).to_csv(report_filepath)
        filename: str = os.path.join(self.dir_path, "results")
        header = not os.path.exists(f"{filename}.csv")
        pd.DataFrame(
            [
                [
                    seed,
                    lr,
                    bs,
                    self.best_val_ce,
                    self.best_val_accuracy,
                    self.best_val_f1,
                    test_ce,
                    test_accuracy,
                    test_f1,
                    self.fine_tuning_time,
                    test_time_taken,
                    report,
                    json.dumps(report_json),
                ]
            ],
            columns=[
                "Seed",
                "Learning Rate",
                "Batch Size",
                "Val CE",
                "Val Accuracy",
                "Val F1",
                "Test CE",
                "Test Accuracy",
                "Test F1",
                "Fine Tuning Time(m)",
                "Test Labeling Time(m)",
                "classification_report",
                "classification_report_json",
            ],
        ).to_csv(f"{filename}.csv", mode="a", header=header)

    def grid_search_bert(self):
        train_dataset, train_str2int, train_int2str = self.load_data(
            self.train_path, self.int2str
        )
        val_dataset, val_str2int, val_int2str = self.load_data(
            self.val_path, self.int2str
        )
        test_dataset, test_str2int, test_int2str = self.load_data(
            self.test_path, self.int2str
        )

        assert train_str2int == val_str2int == test_str2int, f"Labels are mismatching"
        assert train_int2str == val_int2str == test_int2str, f"Labels are mismatching"

        num_labels = len(train_int2str)
        self.set_criterion(train_str2int)

        for seed in self.seeds:
            for lr in self.learning_rates:
                for bs in self.batch_sizes:
                    dataloaders_dict = {
                        "train": DataLoader(train_dataset, batch_size=bs, shuffle=True),
                        "val": DataLoader(val_dataset, batch_size=bs, shuffle=True),
                        "test": DataLoader(test_dataset, batch_size=bs, shuffle=True),
                    }

                    self.set_current_seed(seed)
                    self.set_current_lr(lr)
                    self.set_current_batch_size(bs)

                    model = RobertaForTokenClassification.from_pretrained(
                        self.language_model, num_labels=num_labels
                    ).to(
                        self.device
                    )  # , ignore_mismatched_sizes=True
                    optimizer = optim.AdamW(model.parameters(), lr=lr)

                    self.fine_tune(model, optimizer, dataloaders_dict, train_str2int)
                    self.test(model, optimizer, dataloaders_dict, train_str2int)


def main():
    config_path = sys.argv[1]

    bert_for_ner_setup = BERTForNer(config_path)
    bert_for_ner_setup.grid_search_bert()


if __name__ == "__main__":
    main()
