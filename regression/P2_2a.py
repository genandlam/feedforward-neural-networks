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


NUM_FEATURES = 8
epochs = 100
OPTIMIZER =[5e-7, 1e-7, 5e-9, 1e-9,1e-10]
learning_rate = 1e-9
beta = 0.001
epochs = 100
batch_size = 32
num_neuron = 30


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


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
cvscores = []
all_history=[]
for learning_rate in OPTIMIZER:
    history_model=[]
    for train, test in kfold.split(X_train, y_train):
        model = models.Sequential()
        model.add(Dense(num_neuron,kernel_regularizer=regularizers.l2(beta)
                        ,activation='relu', 
                        input_shape=(NUM_FEATURES,)))
        
        model.add(Dense(1, activation='linear'))

        sgd = optimizers.SGD(lr=learning_rate, momentum=0.00, decay=0.0, nesterov=False)

        model.compile(loss='mean_squared_error', # Cross-entropy
                        optimizer=sgd, # Root Mean Square Propagation
                        metrics=['mse']) # Accuracy performance metric
    # The patience parameter is the amount of epochs to check for improvement
    #    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=20)

        history = model.fit(X_train[train], y_train[train], 
                            batch_size=batch_size,
                            epochs=epochs,
    #                        validation_split=0.2,
                            verbose=0,
                           )

        scores = model.evaluate(X_train[test], y_train[test], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]))
        cvscores.append(scores[1] )
        history_model.append(history)
    total_history= np.asarray(history.history['mean_squared_error'])   

    for history_1 in history_model[:-1]:

        total_history+=np.asarray(history.history['mean_squared_error']) 

    all_history.append(total_history/5) 
 
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    
    plt.figure()
plt.xlabel('Epoch')
plt.ylabel('mean_squared_error')



#plt.plot(history.epoch, (all_history[0]), label='Train Loss 5e-7')
#plt.plot(history.epoch, (all_history[0]), label='Train Loss 1e-7 ')
#plt.plot(history.epoch, (all_history[0]), label='Train Loss 5e-9')
# plt.plot(history.epoch, (all_history[1]), label='Train Loss 1e-9')
 plt.plot(history.epoch, (all_history[2]), label='Train Loss 1e-10')
#plt.savefig('./Pb_2a_1e-10.png')



plt.legend()
plt.show()


