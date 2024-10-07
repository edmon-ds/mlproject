import os
import sys

from dataclasses import dataclass

#from catboost import CatBoostRegressor
#from sklearn.ensemble import(  AdaBoostRegressor , GradientBoostingRegressor , RandomForestRegressor)
#from sklearn.linear_model import LinearRegression
#from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import r2_score

from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet

from src.exception import CustomException
from src.logger import logging 

from src.utils import *

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig

    def initiate_model_trainer(self , train_array , test_array ):
        try:
            logging.info("split training and test input data")
            
            X_train , y_train , X_test , y_test = (train_array[: , :-1] , 
                                                    train_array[: , -1] , 
                                                    test_array[: , :-1] , 
                                                    test_array[: , -1]
                                                    )
            models = {
                        "Decision Tree": DecisionTreeRegressor(),
                        "Elastic Net": ElasticNet(),  # Reemplaza LinearRegression con ElasticNet
                        "XGBRegressor": XGBRegressor(),        
                     }

            params = {
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


            # Adaptar la lista de modelos solo para los que están en params


            model_report:dict = evaluate_models(X_train = X_train , y_train = y_train  ,
                                             X_test = X_test  , y_test = y_test , models = models , param=params)
             
            best_model_name = max(model_report , key = model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info("best found model on both training and testing dataset")
            save_object(file_path = self.model_trainer_config.trained_model_file_path ,obj = best_model)

            predicted = best_model.predict(X_test)
            r2_s = r2_score(y_test , predicted)
            return r2_s
        
        except Exception as e:
            raise CustomException(e , sys)
        
        
            




