# -----------------------------------------------------------------------------
# Predict File
# Author: Xavier Beltran Urbano
# Date Created: 02-01-2024
# -----------------------------------------------------------------------------

import numpy as np
from sklearn.metrics import accuracy_score,cohen_kappa_score
from configuration import Configuration
from keras.models import load_model
from scipy.stats import mode
import pandas as pd

class EfficientNetEnsemble:
    def __init__(self, models_paths, imgPath):
        self.models = []
        self.configs = []
        self.configsTest=[]
        self.imgPath = imgPath
        self.imgSizes = {'EfficientNetB2': (260, 260, 3),
                         'EfficientNetB3': (300, 300, 3),
                         'EfficientNetB4': (380, 380, 3),
                         'EfficientNetB5': (456, 456, 3),
                         'EfficientNetB6': (528, 528, 3),
                         'EfficientNetB7': (600, 600, 3),
                         'DenseNet169': (224, 224, 3),
                         'DenseNet201': (224, 224, 3),
                         'EfficientNetV2B2': (260, 260, 3),
                         'ResNet50V2': (224, 224, 3)
                        }
        
        self.batchSizes = {'EfficientNetB2': 64,
                         'EfficientNetB3': 64,
                         'EfficientNetB4': 64,
                         'EfficientNetB5': 64,
                         'EfficientNetB6': 32,
                         'EfficientNetB7': 32,
                         'DenseNet169': 64,
                         'DenseNet201': 64,
                         'EfficientNetV2B2': 64,
                         'ResNet50V2': 64}
        
        self.models_paths=models_paths
        # Load models and create data generators
        for i, model_name in enumerate(self.models_paths):
            target_size = self.imgSizes[model_name]  # Get the target size from the dictionary
            path=f"/notebooks/results/{model_name}/Best_Model.h5"
            model = self.load_efficientnet_model(path)
            self.models.append(model)
            config = Configuration(pathData=imgPath, targetSize=target_size, batchSize=self.batchSizes[model_name], nClasses=1,modelName=model_name)
            _, valGenerator = config.createAllDataGenerators()
            self.configs.append(valGenerator)
            if i==0:
                config = Configuration(pathData=imgPath, targetSize=target_size, batchSize=self.batchSizes[model_name], nClasses=1,modelName=model_name)
                _, valGenerator_ = config.createAllDataGenerators()
                self.y_true =self.extract_ytrue(valGenerator_) # Extract the Ground Truth

    def load_efficientnet_model(self,path):
        # Load Model
        model=load_model(path, custom_objects={
                    'binary_crossentropy': "binary_crossentropy",
                    'accuracy': "accuracy"})
        return model
    
    def extract_ytrue(self, config):
        # Extract Ground Truth
        y_true = []
        for _, y in config:
            y_true.extend(y)
        return y_true
    
    def predict(self):
        # Make predictions and ensemble different models
        preds = []
        for model, config, model_name in zip(self.models, self.configs,self.models_paths):
            print(f"Predicting with {model_name} model...")
            pred = model.predict(config)
            preds.append(pred)
            print(f"{model_name} predictions complete.")
            
        if len(preds) > 1:
            for type_ens in ['mean', 'max', 'majority']:
                if type_ens == 'mean':
                    ensemble_pred = np.mean(preds, axis=0)[:,0]
                elif type_ens == 'max':
                    ensemble_pred = np.max(np.array(preds), axis=0)[:,0]
                elif type_ens == 'majority':
                    ensemble_pred, _ = mode(preds, axis=0)
                    ensemble_pred = ensemble_pred[0,:,0]  # mode returns a tuple of mode values and counts
                else:
                    raise ValueError("Unsupported ensemble method")
                ensemble_labels = (ensemble_pred > 0.5)
                accuracy = accuracy_score(self.y_true, ensemble_labels)
                kappa = cohen_kappa_score(self.y_true, ensemble_labels)

                print(f" ********** Ensemble {type_ens} **********")
                print(f"Ensemble ({type_ens} method) Accuracy: {accuracy}")
                print(f"Ensemble ({type_ens} method) Kappa Score: {kappa}")
            print("--------------------------------------------------------------")

        else:
            # Single model prediction
            accuracy = accuracy_score(self.y_true, (preds[0] > 0.5))
            kappa = cohen_kappa_score(self.y_true, (preds[0] > 0.5))
            print("--------------------------------------------------------------")
            print(f"Single Model ({self.models_paths[0]}) Accuracy: {accuracy}")
            print(f"Single Model ({self.models_paths[0]}) Kappa Score: {kappa}")
    
                   
    def ReadTestDataset(self):
        # Read Test set to subsequently predict it
        for i, model_name in enumerate(self.models_paths):
            config = Configuration(pathData=imgPath, targetSize=self.imgSizes[model_name], batchSize=self.batchSizes[model_name], nClasses=1,modelName=model_name)
            testGenerator = config.createTestDataGenerators()
            self.configsTest.append(testGenerator)
            if i==0:
                config = Configuration(pathData=imgPath, targetSize=self.imgSizes[model_name], batchSize=self.batchSizes[model_name], nClasses=1,modelName=model_name)
                testGenerator_ = config.createTestDataGenerators()
                file_names=self.extract_ytrue(testGenerator_)
        return file_names
    
    def predictTest(self,type_ensemble='mean'):
        # Predict the test set
        preds = []
        file_names=self.ReadTestDataset()
        for model,config, model_name in zip(self.models,self.configsTest,self.models_paths):
            print(f"Predicting with {model_name} model...")
            pred = model.predict(config)
            preds.append(pred)
            print(f"{model_name} predictions complete.")
            
        if len(preds) > 1:
                if type_ensemble == 'mean':
                    ensemble_pred = np.mean(preds, axis=0)
                elif type_ensemble == 'max':
                    ensemble_pred = np.max(np.array(preds), axis=0)
                elif type_ensemble == 'majority':
                    ensemble_pred, _ = mode(preds, axis=0, keepdims=False)
                    ensemble_pred = ensemble_pred[0]  # mode returns a tuple of mode values and counts
                else:
                    raise ValueError("Unsupported ensemble method")
                predictions = (ensemble_pred > 0.5)                

        else:
            # Single model prediction
            predictions=preds[0] > 0.5
            
        # DataFrame with only predictions
        df_predictions = pd.DataFrame(predictions.astype(int), columns=['Label'])
        predictions_excel_filename = "/notebooks/results/binary_predictions.xlsx"
        df_predictions.to_excel(predictions_excel_filename, index=False, header=False)
        print(f"Labels stored in {predictions_excel_filename}")

        # DataFrame with filenames and predictions
        df_filenames_predictions = pd.DataFrame({'Filename': file_names, 'Label': predictions.flatten().astype(int)})
        filenames_predictions_excel_filename = "/notebooks/results/filenames_binary_predictions.xlsx"
        df_filenames_predictions.to_excel(filenames_predictions_excel_filename, index=False, header=False)
        print(f"Labels and filenames stored in {filenames_predictions_excel_filename}")

    



models_paths=['EfficientNetB4','EfficientNetB5','EfficientNetB6'] # Put here the name of the networks you want to ensemble
imgPath = '/notebooks/data/'
ensemble = EfficientNetEnsemble(models_paths, imgPath)
ensemble.predict()
#ensemble.predictTest(type_ensemble='mean')

