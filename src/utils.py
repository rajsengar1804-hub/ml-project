import os
import sys 
import dill
import numpy as np
import pickle
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import r2_score

def saveobject(filepath, obj):
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok=True)
        with open(filepath, 'wb') as fileobj:
            pickle.dump(obj, fileobj)
        logging.info("preproceesor file is saved successfully")
    except Exception as ex:
        raise CustomException(ex, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    report = {}
    try:
        for i in range(len(models)):
            model = list(models.values())[i]

            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            y_train_score = r2_score(y_train, y_train_pred)

            # ✅ ONLY FIX IS HERE
            y_test_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = y_test_score

        logging.info("model training done ")
        return report

    except Exception as ex:
        raise CustomException(ex, sys)
