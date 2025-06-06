import sys
import pandas as pd
import argparse

from app_logging import logging
from app_exception.exception import AppException

class LoadData:
    '''
    The main functionality of this class is to load the data to the project folder path
    function return data and save it to folder we have assigned
    '''

    def __init__(self):
        pass

    def load_data(self, input_path, output_path):
            
        try:
            logging.info(f"Loading data from the source")
            self.data_path = input_path
            self.data = pd.read_csv(self.data_path, sep=',', encoding='utf-8')

            self.raw_data_path = output_path
            self.data.to_csv(self.raw_data_path, sep=',', encoding='utf-8', index=False)
            logging.info(f"Data Loaded from the source Successfully !!!")
        
        except Exception as e:
            logging.info(
                f"Exception Occurred while loading data from the source -->{e}")
            raise AppException(e, sys) from e



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--input_path', default='data/external/Consignment_pricing.csv', help='Ruta del archivo CSV de entrada usado para entrenar el modelo de predicción')
    args.add_argument('--output_path', default='data/raw/Dataset.csv', help='Ruta de salida para guardar los datos')

    parsed_args = args.parse_args()
    
    LoadData().load_data(input_path=parsed_args.input_path, output_path=parsed_args.output_path)


