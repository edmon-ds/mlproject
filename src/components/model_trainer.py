import os
import sys

from dataclasses import dataclass


from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging 

from src.utils import *

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig
        
        self.models = {
                        "Decision Tree": DecisionTreeRegressor(),
                        "Elastic Net": ElasticNet()  # Reemplaza LinearRegression con ElasticNet
                     }

        self.params = {
                "Decision Tree": {
                                    'max_depth': [None, 5, 10, 15, 20],  # Profundidad máxima
                                },
                "Elastic Net": {
                'alpha': [0.1, 0.5],  # Parámetro de regularización
                'l1_ratio': [0.1, 0.5],  # Proporción de L1 en la regularización
                },
                "XGBRegressor": {
                        'learning_rate': [.1, .01],
                        'n_estimators': [8, 16]
                }
                }
        self.best_model_name = ""
        self.best_model_score = 0

    def initiate_model_trainer(self , train_array , test_array ):
        try:
            logging.info("split training and test input data")
            
            X_train , y_train , X_test , y_test = (train_array[: , :-1] , 
                                                    train_array[: , -1] , 
                                                    test_array[: , :-1] , 
                                                    test_array[: , -1]
                                                    )

            # Adaptar la lista de modelos solo para los que están en params

            model_report:dict = self.evaluate_models(X_train = X_train , y_train = y_train  ,
                                             X_test = X_test  , y_test = y_test ,
                                             models = self.models , param= self.params)
             
            self.best_model_name = max(model_report , key = model_report.get)

            best_model = self.models[self.best_model_name]
            self.best_model_score = model_report[self.best_model_name]
            
            
            if self.best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info("best found model on both training and testing dataset")
            save_object(file_path = self.model_trainer_config.trained_model_file_path ,obj = best_model)
 
        except Exception as e:
            raise CustomException(e , sys)
        
    def evaluate_models(self , X_train, y_train,X_test,y_test,models,param):
        try:
            report = {}

            for i in range(len(list(models))):
                model = list(models.values())[i]
                para = param[list(models.keys())[i]]

                gs = GridSearchCV(model,para,cv=3)
                gs.fit(X_train,y_train)

                model.set_params(**gs.best_params_)
                model.fit(X_train,y_train)

                y_test_pred = model.predict(X_test)

                test_model_score = r2_score(y_test, y_test_pred)

                report[list(models.keys())[i]] = test_model_score

            return report

        except Exception as e:
            raise CustomException(e, sys)