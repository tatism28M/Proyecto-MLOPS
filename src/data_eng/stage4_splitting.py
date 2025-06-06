import argparse
from sklearn.model_selection import train_test_split
import pandas as pd

from app_logging import logging

from data_eng.stage0_loading import GetData
from data_eng.stage3_labeling import FeatureEngineering


class SplitData:
    """
    Simple version of SplitData for early-stage MLOps development. 
    - Splits into train and test
    - Saves to local CSV files
    """

    def __init__(self):
        self.get_data = GetData()
        self.labeling = FeatureEngineering()

    def split_data(self, input_path):
        logging.info("Starting data splitting...")

        # Apply feature engineering (assumes it returns a transformed DataFrame)
        self.data = pd.read_csv(input_path, sep=",")
        logging.info("Feature engineering retrieved.")

        # Split data
        self.train, self.test = train_test_split(self.data, test_size=0.2, random_state=42)
        logging.info("Data split into train and test sets.")

        # Save files locally (basic names for now)
        self.train.to_csv("data/processed/Train_Dataset.csv", sep=",",
                              index=False, encoding="UTF-8")
        self.test.to_csv("data/processed/Test_Dataset.csv", sep=",",
                              index=False, encoding="UTF-8")
        logging.info("Train and test data saved to 'data/train.csv' and 'data/test.csv'.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='data/processed/Processed_Dataset.csv')
    args = parser.parse_args()

    SplitData().split_data(input_path=args.input_path)

