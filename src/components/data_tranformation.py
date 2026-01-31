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
    preprocessor_ob_fie_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def intiate(self):
        '''
        this function is responsible for data transformation 
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scalar", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("onehot", OneHotEncoder()),
                    ("scalar", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor

        except Exception as ex:
            raise CustomException(ex, sys)

    def initiate_data_transformation(self, trainpath, testpath):
        try:
            train_df = pd.read_csv(trainpath)
            test_df = pd.read_csv(testpath)

            logging.info("Read train and test data completed")

            preprocessing_obj = self.intiate()

            target_column = 'math_score'

            input_train_df = train_df.drop(columns=[target_column], axis=1)
            target_train_df = train_df[target_column]

            input_test_df = test_df.drop(columns=[target_column])
            target_test_df = test_df[target_column]

            logging.info(
                "Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_test_df)
            ]

            saveobject(
                filepath=self.data_transformation_config.preprocessor_ob_fie_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_fie_path
            )

        except Exception as ex:
            raise CustomException(ex, sys)
