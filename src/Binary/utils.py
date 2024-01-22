# -----------------------------------------------------------------------------
# Utils class
# Author: Xavier Beltran Urbano
# Date Created: 10-11-2023
# -----------------------------------------------------------------------------

# Import necessary libraries
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
import matplotlib.pyplot as plt
from openpyxl import load_workbook  
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping


class Utils:
    def __init__(self):
        pass

    @staticmethod
    def allCallbacks(networkName):
        # Save weights of each epoch
        os.makedirs(f"/notebooks/results/{networkName}/", exist_ok=True)
        pathWeights=f"/notebooks/results/{networkName}/"
        checkpoint_path = pathWeights+"/epoch-{epoch:02d}.h5"
        model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=False,
        verbose=0)

        # Reduce learning rate
        reduce_lr_callback = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.1,        
        patience=5,       
        min_lr=0.000001,    
        verbose=1)

        # Early stopping callback
        early_stopping_callback = EarlyStopping(
        monitor='val_loss', 
        patience=10,       
        verbose=1)
        return model_checkpoint_callback,reduce_lr_callback,early_stopping_callback

    @staticmethod
    def save_training_plots(history, file_path):
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']

        # Determine the actual number of epochs
        epochs = len(train_loss)

        # Plot training and validation Loss
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epochs + 1), train_loss, label='Training Loss')
        plt.plot(range(1, epochs + 1), val_loss, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        # Plot training and validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(range(1, epochs + 1), accuracy, label='Training Accuracy')
        plt.plot(range(1, epochs + 1), val_accuracy, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()

        # Save the plots to the specified file path
        plt.tight_layout()
        plt.savefig(file_path)
    
    @staticmethod
    def get_predictions(model, generator):
        predicted_classes = []
        true_classes = []
        for batch_data, batch_labels in generator:
            batch_preds = model.predict(batch_data, verbose=0)
            predicted_classes.extend((batch_preds > 0.5).astype(int).flatten())
            true_classes.extend(batch_labels)
        return np.array(true_classes), np.array(predicted_classes)

    @staticmethod
    def calculate_metrics(true_classes, predicted_classes):
        conf_matrix = confusion_matrix(true_classes, predicted_classes)
        accuracy = accuracy_score(true_classes, predicted_classes)
        kappa = cohen_kappa_score(true_classes, predicted_classes)

        # Sensitivity and Specificity
        TP = conf_matrix[1, 1]
        TN = conf_matrix[0, 0]
        FP = conf_matrix[0, 1]
        FN = conf_matrix[1, 0]

        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        return accuracy, kappa, sensitivity, specificity
    
    @staticmethod
    def save_to_excel(metrics, file_path, sheet_name):
        results_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Kappa Score', 'Sensitivity', 'Specificity', 'Loss'],
            'Value': metrics})
        print(file_path)
        if os.path.exists(file_path):
            with pd.ExcelWriter(file_path, engine='openpyxl', mode='a') as writer:
                book = load_workbook(file_path)
                writer.book = book
                writer.sheets = {ws.title: ws for ws in book.worksheets}
                results_df.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                results_df.to_excel(writer, sheet_name=sheet_name, index=False)

    @staticmethod
    def predict_val(path, networkName, validation_generator):
        model = load_model(f'{path}/Best_Model.h5')
        true_classes, predicted_classes = Utils.get_predictions(model, validation_generator)
        metrics = Utils.calculate_metrics(true_classes, predicted_classes)
        # Predict val set
        loss,acc=model.evaluate(validation_generator,verbose=0)
        print(f"\nTest: Dice= {acc}, Loss= {loss}")
        # Convert metrics to a list and append final_loss
        metrics_list = list(metrics) + [loss]
        # Define Excel file and sheet 
        excel_file_path = f'/notebooks/results/{networkName}/Results_Experiments.xlsx'
        sheet_name = networkName
        Utils.save_to_excel(metrics_list, excel_file_path, sheet_name)
        print(f"Metrics saved to {excel_file_path} in the sheet named {sheet_name}")
