from src.components.data_ingestion import * 
from src.components.data_transformation import * 
from src.components.model_trainer import * 

if __name__ =="__main__":
    print("iniciando")
    print()
    
    print("creando ruta de datasets")
    print()
    obj = DataIngestion()
    train_data_path , test_data_path = obj.initiate_data_ingestion()
    
    print("transformando datasets")
    print()
    data_transformation = DataTransformation()
    train_arr , test_arr  = data_transformation.initiate_data_transformation(train_data_path , test_data_path )
    
    print("entrenando varios modelos")
    print()
    
    modeltrainer = ModelTrainer()
    modeltrainer.initiate_model_trainer(train_arr , test_arr)
    print("el modelo ha sido entrenado y elegido")
    print()
    print("nombre y puntuacion del mejor modelo")
    print(f"nombre: {modeltrainer.best_model_name} puntuacion : {modeltrainer.best_model_score}")
   
    print("proceso finalizado")
    print()