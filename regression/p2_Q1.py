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
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker

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

learning_rate = 0.0000001
beta = 0.001
epochs = 500
batch_size = 32
num_neuron = 30



#read and divide data into test and train sets 
cal_housing = np.loadtxt('/media/hdd1/genfyp/housing/CaliforniaHousing/cal_housing.data', delimiter=',')
X_data, Y_data = cal_housing[:,:8], cal_housing[:,-1]
Y_data = (np.asmatrix(Y_data)).transpose()

idx = np.arange(X_data.shape[0])
np.random.shuffle(idx)
X_data, Y_data = X_data[idx], Y_data[idx]
# split into 67% for train and 33% for test
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.30, random_state=seed)


train_data = (X_train- np.mean(X_train, axis=0))/ np.std(X_train, axis=0)
test_data = (X_test- np.mean(X_train, axis=0))/ np.std(X_train, axis=0)

model = models.Sequential()
model.add(Dense(30,kernel_regularizer=regularizers.l2(beta)
                ,activation='relu', 
                input_shape=(NUM_FEATURES,)))
model.add(Dense(1))
model.add(Activation('linear'))

sgd = optimizers.SGD(lr=learning_rate, momentum=0.00, decay=0.0, nesterov=False)

model.compile(loss='mse', # Cross-entropy
                optimizer=sgd, # Root Mean Square Propagation
                metrics=['mae','mse']) # Accuracy performance metric
model.summary()

history = model.fit(train_data, y_train, 
                    batch_size=batch_size, epochs=epochs,
                    validation_split=0.2,
                    verbose=0)
scores = model.evaluate(X_train, y_train, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]))
print("%s: %.2f%%" % (model.metrics_names[2], scores[2]))

plt.xlabel('Epoch')
plt.ylabel('Train error')
plt.plot(history.history['mean_squared_error'], label='Train error')
#plt.plot(history.history['mean_absolute_error'], label='Train Loss')
#plt.plot((history.history['val_mean_absolute_error']),label = 'Val loss')
plt.legend()
#plt.savefig('./Pb_1a_mean_squared_error.png')
plt.show()


test_50_X=X_test[:50]
test_predictions = model.predict(X_test[:50]).flatten()
y_predict = (np.asmatrix(test_predictions)).transpose()
test_50_y=y_test[:50]



fig = plt.figure(2)
ax = fig.gca(projection = '3d')
ax.scatter(test_50_X[:,0], test_50_X[:,1], y_predict[:,0], color='red', marker='.', label='predicted')
ax.scatter(test_50_X[:,0], test_50_X[:,1], test_50_y[:,0], color='blue', marker='x', label='targets')

ax.set_title('Targets and Predictions')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
ax.legend()
plt.show()
#plt.savefig('./Pb_1b.png')
