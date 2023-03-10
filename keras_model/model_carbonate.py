import tensorflow as tf
# from tensorflow.keras.optimizers.legacy import Adam
# from tensorflow.keras.optimizers.legacy import Adam
import tensorflow.keras.backend as K
tf.compat.v1.disable_eager_execution()
import tensorflow.python.keras.backend as K
import numpy as np
import math
import sys
import os
import data_format
import shutil

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Add
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from tensorflow.keras.optimizers.legacy import Adam

# from tensorflow.keras.optimizer_v1 import Adam
from tensorflow.keras.initializers import glorot_normal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

#Logging 
import logging

#tf.enable_eager_execution()
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

checkpoint_path = "../checkpoints/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
# oldcheckpoint_dir = os.path.dirname(oldcheckpoint_path)
data_path = "../combined.h5"
# oldmodel_path = "../deep_motion_planner/models/modelstore/saved_model11"

VERBOSE = True
TRAIN = False
SAVE = False

#TRAINING PARAMS
batches = 1
learning_rate = 0.0001
epoch = 50
loadmodel = False

def createResBlocks(X, filter_size, initializer, regularizer):
    """ Creates the residual part of the model (like 80% of the entire model)
        Including all skip connections
        X - input tensor
        filter_size - size of kernel for convolutions
    """
    #Save input Tensor
    X_shortcut = X

    #First Component of main path
    X = Conv2D(filter_size, (1,3), padding='same', kernel_initializer=initializer,\
            kernel_regularizer=regularizer)(X)
    X = BatchNormalization()(X)

    #Second Component of main path
    X = Conv2D(filter_size, (1,3), padding='same', kernel_initializer=initializer,\
            kernel_regularizer=regularizer)(X)
    X = BatchNormalization()(X)

    #Add first shortcut value to main path and pass it through ReLU
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    #Save second shortcut
    X_shortcut1 = X

    #Third Component of main path
    X = Conv2D(filter_size, (1,3), padding='same', kernel_initializer=initializer,\
            kernel_regularizer=regularizer)(X)
    X = BatchNormalization()(X)

    #Fourth Component of main path
    X = Conv2D(filter_size, (1,3), padding='same', kernel_initializer=initializer,\
            kernel_regularizer=regularizer)(X)
    X = BatchNormalization()(X)

    #Add second shortcut value to main path and pass it through ReLU
    X = Add()([X, X_shortcut1])
    X = Activation('relu')(X)

    return X


def createModel():
    filter_size = 64
    initializer = tf.keras.initializers.glorot_normal()
    regularizer = tf.keras.regularizers.L1L2(l1=0.001)
    #**** Convolutional Network ****#
    #Create First Block
    X_input = Input(shape=(1,1080,1))
    X = Conv2D(filter_size, (1,7), 3, padding='same',\
            kernel_initializer=initializer, kernel_regularizer=regularizer)(X_input)
    X = BatchNormalization()(X)
    X = MaxPooling2D((1,3), (1,3), padding='same')(X)
    
    #Create Residual Blocks
    X = createResBlocks(X, filter_size, initializer, regularizer)
    
    #Add last pooling layer of CNN
    X = AveragePooling2D((1,3), (1,3), padding='same')(X)
    X = Flatten()(X)
    X = Model(inputs=X_input, outputs=X)

    #**** Fully Connected Layers ****#    
    #Add a secondary input to hidden layer for target info
    Y_in = Input(shape=(3))
    #Y = Lambda(lambda x: x)(Y_in)
    #Y = Model(inputs=Y_in, outputs=Y) 
    
    #Combine new input with output of CNN #Y.output]
    combined = concatenate([X.output, Y_in], axis=1)

    #Add Fully Connected Layers
    Y = Dense(1024, activation=tf.nn.relu)(combined) 
    Y = Dropout(0.5)(Y) #nick is adding dropout to these three layers
    Y = Dense(1024, activation=tf.nn.relu)(Y)
    Y = Dropout(0.5)(Y) #nick is adding dropout to these three layers
    Y = Dense(512)(Y)
    Y = Dense(2)(Y)

    model = Model(inputs=[X_input, Y_in], outputs=Y)

    return model


def resetModel():
    """ Randomly sets weights for the entire model
        Exits program when completed
    """
    model = createModel()
    model.compile(loss='mean_squared_error',optimizer=Adam(0.1))
    
    initial_weights = model.get_weights()
    initializer = tf.keras.initializers.glorot_uniform()
    k_eval = lambda placeholder: placeholder.eval(session=K.get_session())
    
    new_weights = [k_eval(initializer(w.shape)) for w in initial_weights]
    model.set_weights(new_weights)
    print('All weights have been reset to random values.')


    '''added by nick, delete previous checkpoint'''
    print("checking for file")
    if (os.path.exists(checkpoint_dir)):
        # for file in os.listdir(checkpoint_dir):
        #     os.remove(file)
        shutil.rmtree(checkpoint_dir)
        print("previous checkpoint removed")

    else:
        print("no previous checkpoint")

    exit(0)


def initModel():
    model = createModel()
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate), metrics=['accuracy'])

    return model

#++++++++++++++++++++++++++++++++EXECUTION POINT++++++++++++++++++++++++++++++++++++#
print('\n')
#Process cmd line args
if len(sys.argv) > 1:
    for i in sys.argv:
        if (i == '-v'):
            VERBOSE = True
        elif (i == 'reset'):
            resetModel()
        elif (i == 'train'):
            TRAIN = True
        elif (i == 'eval'):
            pass
        elif (i == 'save'):
            SAVE = True
else:
    print('USAGE: python3 model.py <verbosity> <function>')
    print('Verbose mode: -v')
    print('Functions: \'train\' | \'reset\'')
    exit(0)

if VERBOSE: print('Preparing Model & Data...')
model = createModel()

if VERBOSE: print('Compiling Model...')
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate), metrics=['accuracy'])

#If checkpoint file exists, load weights
if loadmodel:
    print("loading model 11")
    model = load_model(oldmodel_path)
elif (os.path.exists(checkpoint_dir)):
    model.load_weights(checkpoint_path)
    print("checkpoint existedDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")
    # print(model.load_weights(checkpoint_path))
    # print("True")
else:
    print("no checkpoint :00000000000")
    # print(os.listdir(checkpoint_dir))

#Create callback for saving/restoring model weights
cp_callback = ModelCheckpoint(filepath=checkpoint_path, 
        save_best_only=True, save_weights_only=True, verbose=1, monitor='val_accuracy')
es = EarlyStopping(monitor='val_accuracy', patience=15)

if VERBOSE: print('Preparing Input Data...')
dh = data_format.DataHandler(data_path)#'./data/evaluation.h5')
lidarTrain, lidarTest, targetTrain, targetTest,\
                    labelTrain, labelTest, = dh.get_data()
num_points = dh.get_training_points()

if (TRAIN):
    if VERBOSE: print('Initiating Training...')
    history = model.fit([lidarTrain, targetTrain], labelTrain, 
        validation_data=([lidarTest, targetTest], labelTest), epochs=epoch, batch_size=batches,
        callbacks=[cp_callback, es])
    if VERBOSE: print('Training Completed.')
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

if (SAVE):
    model.save("../deep_motion_planner/models/saved_model")
    np.save("loss",history.history['loss'])
    np.save("val_loss",history.history['val_loss'])
    print("model saved")
#     tf.keras.experimental.export_saved_model(model, "./src/keras_model/model_backup/test_save")

# Evaluate Accuracy - Print results
# metrics = model.evaluate([lidarTest, targetTest], labelTest)
# print(metrics)

