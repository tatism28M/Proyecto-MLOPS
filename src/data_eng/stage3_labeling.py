import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from app_logging import logging
from app_exception.exception import AppException


class FeatureEngineering:
    '''
    This machine learning pipeline this is for feature engineering technique 
    like outlier handling,column transformation using one hot and label encoding 
    function return and save data as 'processed_data' folder
    '''

    def __init__(self):
        pass 
        
    def data_(self, input_path):
        try:
            logging.info("'data_' FUNCTION STARTED")
            self.data_path = input_path 
            self.data = pd.read_csv(self.data_path, sep=',', encoding='utf-8')
            logging.info("Data loaded successfully")
            return self.data
        except Exception as e:
            logging.info(
                 "Exception occurred while loading the data" + str(e))
            logging.info(
                 "Failed to load the data please check your code and run")
            raise AppException(e, sys) from e
    
    
    # outlier detection
    def outlier_detection(self, data, colname):
        self.data = data[data[colname] <= (
            data[colname].mean()+3*data[colname].std())]
        return self.data
    
    
    def remove_outliers(self, input_path):
        try:
            logging.info( "'remove_outliers' FUNCTION STARTED")
            self.data = self.data_(input_path)
            self.data0 = self.outlier_detection(self.data, "line_item_value")
            self.data1 = self.outlier_detection(
                self.data0, "unit_of_measure_(per_pack)")
            self.data2 = self.outlier_detection(self.data1, "pack_price")
            self.data3 = self.outlier_detection(self.data2, "unit_price")
            # self.data4=self.outlier_detection(self.data3,"days_to_process")
            self.data = self.data3
            logging.info(
                 "removed outliers function compiled successfully")
            return self.data
        except Exception as e:
            logging.info(
                 "Exception occured in remove_outliers method"+str(e))
            logging.info( "Error occured while removing outliers")
            raise AppException(e, sys) from e



    def trans_freight_cost(self, x):
            if x.find("See") != -1:
                return np.nan
            elif x == "Freight Included in Commodity Cost" or x == "Invoiced Separately":
                return 0
            else:
                return x

    def freight_cost_transform(self, input_path):
        try:
            logging.info(
                 "'freight_cost_transform' FUNCTION STARTED")
            self.data = self.remove_outliers(input_path)
            
            self.data = self.data.copy()
            self.data["freight_cost_(usd)"] = self.data["freight_cost_(usd)"].apply(self.trans_freight_cost)
            
            self.data = self.data.copy()
            self.median_value = self.data["freight_cost_(usd)"].median()

            self.data = self.data.copy()
            self.data["freight_cost_(usd)"] = self.data["freight_cost_(usd)"].replace(
                np.nan, self.median_value)
            
            self.data = self.data.copy()
            self.data["freight_cost_(usd)"] = self.data["freight_cost_(usd)"].astype(
                float)
            logging.info("freight_cost_transform function compiled successfully")
     
            return self.data
        except Exception as e:
            logging.info(
                 "Exception occurred while compiling the code" + str(e))
            logging.info(
                 "Failed to execute the code please check your code and run")
            raise AppException(e, sys) from e


    def feature_engineering(self, input_path):
        try:
            logging.info( "'feature_engineering' FUNCTION STARTED")
            lb=LabelEncoder()
            lb=LabelEncoder()
            self.data = self.freight_cost_transform(input_path)
            self.data.drop("pq_#",axis=1,inplace=True)
            self.data["po_/_so_#"]=pd.get_dummies(self.data["po_/_so_#"],drop_first=True,dtype=int)
            self.data["asn/dn_#"]=pd.get_dummies(self.data["asn/dn_#"],drop_first=True,dtype=int)
            
            self.data["country"]=lb.fit_transform(self.data["country"])
            self.data["fulfill_via"]=pd.get_dummies(self.data["fulfill_via"],drop_first=True,dtype=int)
            self.data["vendor_inco_term"]=lb.fit_transform(self.data["vendor_inco_term"])
            
            self.data = self.data.copy()
            self.data["sub_classification"]=lb.fit_transform(self.data["sub_classification"])

            self.data = self.data.copy()
            self.data["first_line_designation"]=pd.get_dummies(self.data["first_line_designation"],drop_first=True,dtype=int)

            self.data["shipment_mode"] = lb.fit_transform(self.data["shipment_mode"])            
            logging.info(
                 "feature engineering function compiled successfully")
            return self.data
            # [data for data in self.data if self.data[data].dtypes=="O"]
        except Exception as e:
            logging.info(
                 "Exception occurred while compiling the code" + str(e))
            logging.info(
                 "Failed to execute the code please check your code and run")
            raise AppException(e, sys) from e

    def final_data(self,input_path, output_path):
        try:
            
            logging.info( "'data' FUNCTION STARTED")
            self.finaldata = self.feature_engineering(input_path)
            self.data.drop("Unnamed: 0", axis=1, inplace=True)
            self.data.to_csv(output_path, index=False)
            logging.info( "Final Data for prediction successfully created")
        except Exception as e:
            logging.info(
                 "Exception occurred while compiling the code" + str(e))
            logging.info(
                 "Failed to execute the code please check your code and run")
            raise AppException(e, sys) from e

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='data/interim/Cleaned_Dataset.csv')
    parser.add_argument('--output_path', default='data/processed/Processed_Dataset.csv')
    args = parser.parse_args()

    data = FeatureEngineering().final_data(input_path=args.input_path, output_path=args.output_path)
