# -----------------------------------------------------------------------------
# Main Binary file
# Author:  Xavier Beltran Urbano 
# Date Created: 31-10-2023
# -----------------------------------------------------------------------------

from utils import Utils
from network import *
import os
from metrics import *
from keras.optimizers import Adam, SGD
from configuration import Configuration
from keras import backend as K




def run_program(config,networkName, params, crossVal=5):
    # Clear any existing TensorFlow session
    K.clear_session()
    utils = Utils()
    # Generate the IDs for train and val
    trainGenerator, valGenerator=config.createAllDataGenerators()

    # Define model and compile
    network=ClassifNetwork(config.targetSize)
    model = network.get_model()
    model.summary()
    model.compile(optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=["accuracy"])
    
    with tf.device('/GPU:0'):
        # Train the model
        model_checkpoint_callback,reduce_lr_callback,early_stopping_callback = utils.allCallbacks(networkName)
        epochs = 300
        history = model.fit(trainGenerator, validation_data=valGenerator, verbose=1,epochs=epochs,
                           callbacks=[model_checkpoint_callback, reduce_lr_callback, early_stopping_callback])

        model.save(f"/notebooks/results/{networkName}/Best_Model.h5")
        # Plot the results and save the image
        utils.predict_val(f"/notebooks/results/{networkName}/",networkName,valGenerator)


if __name__ == "__main__":
    imgPath = '/notebooks/data/'
    # Nertwork Unet
    networkName="DenseNet169"
    
    # Parameters of the training
    params={
        'pathData':imgPath,
        'targetSize':(224,224,3),
        'batchSize':128,
        'nClasses':1
    }
    
    # Create folder for this experiment
    networkName=networkName
    os.makedirs(f"/notebooks/results/{networkName}", exist_ok=True)
    
    # Configuration of the experiment
    config=Configuration(**params)
    
    #Run experiment
    run_program(config,networkName,params)
