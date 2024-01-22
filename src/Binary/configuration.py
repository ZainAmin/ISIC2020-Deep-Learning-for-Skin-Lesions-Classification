# -----------------------------------------------------------------------------
# Configuration Class
# Author: Xavier Beltran Urbano
# Date Created: 10-12-2023
# -----------------------------------------------------------------------------

import os
import random
from dataGenerator import DataGenerator

class Configuration():
    def __init__(self,pathData,targetSize,batchSize,nClasses,modelName):
        self.pathData=pathData
        self.targetSize=targetSize
        self.batchSize=batchSize
        self.nClasses=nClasses
        self.modelName=modelName

    def readDataset(self):
        # Read the image IDs from the train folder
        nevus_ids = [filename.split('.')[0] for filename in os.listdir(self.pathData + "train/nevus/")]
        others_ids = [filename.split('.')[0] for filename in os.listdir(self.pathData + "train/others/")]
        random.shuffle(nevus_ids), random.shuffle(others_ids)

        val_nevus = [filename.split('.')[0] for filename in os.listdir(self.pathData + "val/nevus/")]
        val_others = [filename.split('.')[0] for filename in os.listdir(self.pathData + "val/others/")]

        train_ids = nevus_ids + others_ids
        val_ids = val_nevus + val_others

        # Shuffle the combined training and validation lists
        random.shuffle(train_ids), random.shuffle(val_ids)
        return train_ids, val_ids
    
    def readTestDataset(self):
        # Read the image IDs from the test folder
        test_ids = sorted([filename.split('.')[0] for filename in os.listdir(self.pathData + "test/")])
        return test_ids


    def createDataGenerator(self, list_IDS, dataAugmentation, setType,testSet, shuffle=True):
        # Create the Data Generator
        data_generator = DataGenerator(
            image_directory=self.pathData + setType,
            list_IDS=list_IDS,
            batch_size=self.batchSize,
            target_size=self.targetSize,
            data_augmentation=dataAugmentation,
            modelName=self.modelName,
            shuffle=shuffle,
            testSet=testSet)
        return data_generator

    def createAllDataGenerators(self):
        # Read IDs
        train_ids, val_ids = self.readDataset()
        # Create the all the Data Generators
        train_generator = self.createDataGenerator(train_ids, dataAugmentation=True, setType='train/', shuffle=True,testSet=False)
        validation_generator = self.createDataGenerator(val_ids, dataAugmentation=False, setType='val/', shuffle=False,testSet=False)  # Set shuffle to False for validation
        return train_generator, validation_generator  
    
    def createTestDataGenerators(self):
        # Read test IDs
        test_ids = self.readTestDataset()
        # Create the Data Generators
        test_generator = self.createDataGenerator(test_ids, dataAugmentation=False, setType='test/', shuffle=False, testSet=True)
        return test_generator  
    
