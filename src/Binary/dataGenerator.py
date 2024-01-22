# -----------------------------------------------------------------------------
# Data Generator Class
# Author: Xavier Beltran Urbano
# Date Created: 10-11-2023
# -----------------------------------------------------------------------------

# Import necessary libraries
import os
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from skin_preprocessing import Preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Different types of preprocess_input for each pretrained model
def preprocess_input(img, modelName):
    if 'EfficientNet' in modelName:
        from tensorflow.keras.applications.efficientnet import preprocess_input
    elif 'DenseNet' in modelName:
        from tensorflow.keras.applications.densenet import preprocess_input
    elif 'ResNet50V2' in modelName:
        from tensorflow.keras.applications.resnet_v2 import preprocess_input
    elif 'EfficientNetB2V2' in modelName:
        from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
    else:
        raise ValueError("Unknown modelName")

    return preprocess_input(img)



class DataGenerator(Sequence):
    def __init__(self, image_directory, list_IDS, batch_size, target_size, data_augmentation,
                 modelName,testSet=False,shuffle=True):
        self.image_directory = image_directory
        self.batch_size = batch_size
        self.target_size = target_size
        self.modelName=modelName
        self.shuffle = shuffle
        self.testSet=testSet
        if  self.testSet==False: # Train and Val
            image_filenames_nevus = [filename for filename in os.listdir(image_directory+"nevus") if
                                           filename.endswith('.jpg') and filename.split('.')[0] in list_IDS]
            image_filenames_others= [filename for filename in os.listdir(image_directory + "others") if
                                     filename.endswith('.jpg') and filename.split('.')[0] in list_IDS]
            self.image_filenames=sorted(image_filenames_nevus+image_filenames_others)
        else: # Test
            image_filenames= [filename for filename in os.listdir(image_directory) if filename.endswith('.jpg') and filename.split('.')[0] in list_IDS]
            self.image_filenames=sorted(image_filenames)
            
        # Calculate the number of batches
        self.num_batches = len(self.image_filenames) // self.batch_size

        # Initialize indices for data shuffling
        self.indices = np.arange(len(self.image_filenames))

        # Preprocessing class
        self.preprocessing = Preprocessing(target_size=target_size)

        # Shuffle the indices
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        # Data Augmentation setup
        if data_augmentation:
            self.augmentor = ImageDataGenerator(
                rotation_range=90,       # Degree range for random rotations
                width_shift_range=0.2,   # Range (as a fraction of total width) for horizontal shifts
                height_shift_range=0.2,  # Range (as a fraction of total height) for vertical shifts
                shear_range=0.2,         # Shear angle in counter-clockwise direction
                zoom_range=0.2,          # Range for random zoom
                horizontal_flip=True,    # Randomly flip inputs horizontally
                brightness_range=[0.8,1.2],  # Range for brightness adjustment
                fill_mode='nearest'      # Points outside boundaries are filled according to the given mode
            )
        else:
            self.augmentor = None

        
    def __len__(self):
        """
        Get the number of batches per epoch.
        """
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, index):
        """
        Generate a batch of data.

        Args:
            index (int): Index of the batch.

        Returns:
            batch_images (np.ndarray): Batch of preprocessed images.
            batch_masks (np.ndarray): Batch of corresponding masks.
        """
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_image_filenames = [self.image_filenames[i] for i in batch_indices]

        # Load and preprocess batch images and masks
        batch_images = batch_gts =[]
        for image_filename in batch_image_filenames:
            if self.testSet==False:
                if "nev"in image_filename:
                    image_path = os.path.join(self.image_directory, "nevus",image_filename)
                    gt = 0
                else:
                    image_path = os.path.join(self.image_directory,"others", image_filename)
                    gt = 1
            else:
                image_path = os.path.join(self.image_directory,image_filename)
                gt = image_filename  # instead of returning the gt, for test we return the filename
                    
            img = load_img(image_path, target_size=self.target_size, color_mode='rgb')
            img = img_to_array(img)
            
            batch_images.append(img)
            batch_gts.append(gt)
        
        batch_images = self.preprocessing.extractROI_batch(batch_images) # Extract ROI
        batch_images = np.array(batch_images, dtype=np.float32)

        for i in range((batch_images.shape[0])):
            # Apply data augmentation
            if self.augmentor:
                batch_images[i,...]= preprocess_input(self.augmentor.random_transform(batch_images[i,...]),self.modelName)
            else:
                batch_images[i,...]= preprocess_input(batch_images[i,...],self.modelName)

        return batch_images,np.asarray(batch_gts)

    def on_epoch_end(self):
        """
        Shuffle the data indices at the end of each epoch if required.
        """
        if self.shuffle:
            np.random.shuffle(self.indices)
