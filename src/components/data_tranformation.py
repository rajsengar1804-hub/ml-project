import os
import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import saveobject

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def intiate(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"
            ]

            num_pipeline = Pipeline(
                steps=[("imputer", SimpleImputer(strategy='median')), ("scaler", StandardScaler())]
            )

            cat_pipeline = Pipeline(
                steps=[("imputer", SimpleImputer(strategy='most_frequent')),
                       ("onehot", OneHotEncoder()),
                       ("scaler", StandardScaler(with_mean=False))]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor

        except Exception as ex:
            raise CustomException(ex, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            preprocessing_obj = self.intiate()
            target_column = "math_score"

            X_train = train_df.drop(columns=[target_column], axis=1)
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column], axis=1)
            y_test = test_df[target_column]

            logging.info("Applying preprocessing object on training and testing data")

            X_train_scaled = preprocessing_obj.fit_transform(X_train)
            X_test_scaled = preprocessing_obj.transform(X_test)

            train_arr = np.c_[X_train_scaled, np.array(y_train)]
            test_arr = np.c_[X_test_scaled, np.array(y_test)]

            saveobject(
                filepath=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as ex:
            raise CustomException(ex, sys)
