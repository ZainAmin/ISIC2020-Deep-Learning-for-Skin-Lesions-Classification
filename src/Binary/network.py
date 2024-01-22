# -----------------------------------------------------------------------------
# Network class
# Author: Xavier Beltran Urbano
# Date Created: 10-11-2023
# -----------------------------------------------------------------------------

# Import necessary libraries
from keras.models import Model
from keras.layers import Conv2DTranspose,concatenate,Flatten, Activation,GlobalAveragePooling2D, Dropout, Dense,Input, Conv2D, BatchNormalization,MaxPooling2D,Concatenate
from metrics import *
from keras.applications import EfficientNetB0,EfficientNetB2,EfficientNetB4,EfficientNetB3,EfficientNetB5,EfficientNetB6,EfficientNetB7,ResNet50V2,EfficientNetV2M,EfficientNetV2B2,DenseNet201,DenseNet169
from tensorflow.keras import backend as K


    
class ClassifNetwork:
    def __init__(self,target_size):
        self.target_size=target_size
    def get_model(self):
        
        K.set_image_data_format('channels_last')
        base_model = EfficientNetB5(weights="imagenet", include_top=False, input_shape=self.target_size)

        base_model.trainable = True # We do not freeze any layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)
        x = Dense(1, activation='sigmoid')(x)

        # Create the model
        model_classification = Model(inputs=base_model.input, outputs=x)
        return model_classification
    

