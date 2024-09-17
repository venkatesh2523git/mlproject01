import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomeException
from src.logger import logging
from src.utils import save_object

#A data class is a special type of class that is designed to store data. 
#In Python, data classes are created using the @dataclass decorator. 
#They are meant to be used for simple, immutable data structures that have no behavior

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        """
         This function is responsible for data transformations
        """
        try:
            num_columns = ['writing score', 'reading score']
            cat_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course",
            ]
            num_pipeline = Pipeline(
                steps=
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
                )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one-hot", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info(f"Numerical columns: {num_columns}")
            logging.info(f"Categorical columns: {cat_columns}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, num_columns),
                    ("cat_pipeline", cat_pipeline, cat_columns),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomeException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math score"
            num_columns = ['writing score', 'reading score']

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # 'np.r_'  --> it concatenates arrays along first axis
            # 'np.c_' --> it concatenates arrays along second axis
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_test_df)
            ]

            logging.info("Saved preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )



        except Exception as e:
            raise CustomeException(e, sys)

        











