import os
import sys 
import dill
import numpy as np
import pickle
from src.logger import logging
from src.exception import CustomException
def saveobject(filepath,obj):
    try :
        dir_path=os.path.dirname(filepath)
        os.makedirs(dir_path,exist_ok=True)
        with open(filepath,'wb') as fileobj:
            pickle.dump(obj,fileobj)
        logging.info("preproceesor file is saved successfully")
    except Exception as ex:
        raise CustomException(ex,sys)
