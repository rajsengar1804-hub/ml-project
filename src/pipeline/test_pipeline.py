import os 
import sys 
from src.logger import logging
from src.exception import CustomException
import pandas as pd 
import numpy as np
from src.utils import load_object

class CustomData:
     def __init__(  self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score
     def get_data_as_data_frame(self):
         try :
             CustomData_dict={
                  "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
             }
             return pd.DataFrame(CustomData_dict)
             logging.info("input data converted into data frame")

         except  Exception as ex:
             raise CustomException(ex,sys)
class PredictPipeline:
    def __init__(self):
        pass 
    def predict(self,features):
        try :
            model_path=os.path.join("artifacts","Model.pkl")      
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            model=load_object(filepath=model_path)
            preprocessor=load_object(filepath=preprocessor_path)
            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)

            return pred 

        except Exception as ex :
            raise CustomException(ex,sys)

    
        