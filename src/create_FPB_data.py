import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from src.config import SEEDS
from pathlib import Path
import sys
import datasets
import zipfile
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def encode(label_word):
    if label_word == "positive":
        return 0
    elif label_word == "negative":
        return 1
    elif label_word == "neutral":
        return 2
    else:
        return -1


def get_FPB_dataset():
    logger.info(f"Getting FPB from HuggingFace datasets")
    # Load the dataset
    dataset = datasets.load_dataset("financial_phrasebank")
    # Path to save the dataset as a zip file
    zip_path = DATA_DIRECTORY / "FinancialPhraseBank-v1.0.zip"
    # Save the dataset to a zip file
    logger.info(f"Saving FPB to {zip_path}")
    dataset.save_to_disk(zip_path)
    # Unzip the file
    logger.info(f"Unzipping {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall("ExtractedFinancialPhraseBank")


def process_data():
    FPB_DIRECTORY = DATA_DIRECTORY / "ExtractedFinancialPhraseBank"
    FPB_DIRECTORY.mkdir(parents=True, exist_ok=True)
    logger.info(f"Processing the FPB data into {FPB_DIRECTORY}")
    df = pd.read_csv(
        FPB_DIRECTORY / "Sentences_AllAgree.txt",
        sep="@",
        encoding="latin1",
        names=["sentence", "label"],
    )
    df["label"] = df["label"].apply(lambda x: encode(x))

    # TODO: have it build to the directories when reused i.e. numclaim_detection
    TRAIN_DIRECTORY = DATA_DIRECTORY / "sentiment_analysis" / "train"
    TRAIN_DIRECTORY.mkdir(parents=True, exist_ok=True)
    TEST_DIRECTORY = DATA_DIRECTORY / "sentiment_analysis" / "test"
    TEST_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for seed in tqdm(SEEDS):
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)
        train_df.to_excel(
            TRAIN_DIRECTORY / f"FPB-sentiment-analysis-allagree-train-{seed}.xlsx",
            index=False,
        )
        test_df.to_excel(
            TEST_DIRECTORY / f"FPB-sentiment-analysis-allagree-test-{seed}.xlsx",
            index=False,
        )


if __name__ == "__main__":
    ROOT_DIRECTORY = Path(__file__).resolve().parent.parent
    if str(ROOT_DIRECTORY) not in sys.path:
        sys.path.insert(0, str(ROOT_DIRECTORY))
    DATA_DIRECTORY = ROOT_DIRECTORY / "data"
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Building the FinancialPhraseBank dataset in data directory {DATA_DIRECTORY}"
    )
    get_FPB_dataset()
    process_data()
