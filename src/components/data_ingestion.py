import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_path: str = os.path.join("artifacts", "train.csv")
    test_path: str = os.path.join("artifacts", "test.csv")
    raw_path: str = os.path.join("artifacts", "raw_data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initate(self):
        logging.info("Entered the data ingestion method or component")
        try:
            logging.info("read the data from the file")

            df = pd.read_csv(r"src\components\notebook\student.csv")

            os.makedirs(
                os.path.dirname(self.ingestion_config.train_path),
                exist_ok=True
            )

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42
            )

            train_set.to_csv(
                self.ingestion_config.train_path,
                index=False,
                header=True
            )

            test_set.to_csv(
                self.ingestion_config.test_path,
                index=False,
                header=True
            )

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_path,
                self.ingestion_config.test_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initate()
    print("my data = ",train_data)
