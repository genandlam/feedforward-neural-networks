import pandas as pd
import numpy as np 
import os 
import tensorflow as tf
import numpy as np
import pylab as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Conv1D, Activation, \
                         Dropout, MaxPooling1D, GlobalAveragePooling1D, \
                         GlobalMaxPooling1D, Lambda, Concatenate, Dense, regularizers,Flatten 
from keras import backend as K        
from keras import optimizers, activations, models ,callbacks
from keras import callbacks
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="1"
seed = 7
np.random.seed(seed)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)
print("Using GPU: ", K.tensorflow_backend._get_available_gpus())

K.set_image_dim_ordering('tf')


#read and divide data into test and train sets 
cal_housing = np.loadtxt('/media/hdd1/genfyp/housing/CaliforniaHousing/cal_housing.data', delimiter=',')
X_data, Y_data = cal_housing[:,:8], cal_housing[:,-1]
# Y_data = (np.asmatrix(Y_data)).transpose()

idx = np.arange(X_data.shape[0])
np.random.shuffle(idx)
X_data, Y_data = X_data[idx], Y_data[idx]
# split into 70% for train and 30% for test
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.30, random_state=seed)

train_data = (X_train- np.mean(X_train, axis=0))/ np.std(X_train, axis=0)
test_data = (X_test- np.mean(X_train, axis=0))/ np.std(X_train, axis=0)

NUM_FEATURES = 8
learning_rate = 1e-9
beta = 0.001
epochs = 500
batch_size = 32
#change to the best performing
num_neuron= 100
n= 100


def build_model(learning_rate):
    model = models.Sequential()
    model.add(Dense(100,kernel_regularizer=regularizers.l2(beta)
                    ,activation='relu', 
                    input_shape=(NUM_FEATURES,)))
    model.add(Dense(1))
    model.add(Activation('linear'))

    sgd = optimizers.SGD(lr=learning_rate, momentum=0.00, decay=0.0, nesterov=False)

    model.compile(loss='mean_squared_error', # Cross-entropy
                    optimizer=sgd, # Root Mean Square Propagation
                    metrics=['mse','mae']) # Accuracy performance metric
#    model.summary()

    history = model.fit(train_data, y_train, 
                        batch_size=batch_size, epochs=epochs,
                        validation_split=0.2,
                        verbose=0)
    scores = model.evaluate(test_data, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]))

    
    return model,history

 model,history=build_model(learning_rate=1e-10)