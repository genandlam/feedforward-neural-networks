#
# Project 1, starter code part a
#

import time
import math
import tensorflow as tf
import numpy as np
import pylab as plt

#trainX = first layer x input
#train_Y = column vector of output
#trainY = vector arranged, 6 columns 

#x input layer matrix, 1 row n columns
#y_ output layer matrix, 1 row n columns with probabilities

# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)
		

NUM_FEATURES = 36
NUM_CLASSES = 6

learning_rate = 0.01
beta = 0.000001
epochs = 1000
batch_size = 16
num_neurons_array = [5,10,15,20,25]
seed = 10
np.random.seed(seed)

#read train data
train_input = np.loadtxt('sat_train.txt',delimiter=' ')
trainX, train_Y = train_input[:,:36], train_input[:,-1].astype(int)
trainX = scale(trainX, np.min(trainX, axis=0), np.max(trainX, axis=0))
train_Y[train_Y == 7] = 6

trainY = np.zeros((train_Y.shape[0], NUM_CLASSES))
trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1 #one hot matrix

#read test data
test_input = np.loadtxt('sat_test.txt',delimiter=' ')
testX, test_Y = test_input[:,:36], test_input[:,-1].astype(int)
testX = scale(testX, np.min(testX, axis=0), np.max(testX, axis=0))
test_Y[test_Y == 7] = 6

testY = np.zeros((train_Y.shape[0], NUM_CLASSES))
testY[np.arange(test_Y.shape[0]), test_Y-1] = 1 #one hot matrix

# experiment with small datasets
# trainX = trainX[:100]
# trainY = trainY[:100]

n = trainX.shape[0]



batch_single_epoch = []
batch_time_set = []
batch_train_acc = {}
batch_test_acc = []
batch_train_err = {}
batch_test_err = []
batch_train_cross_entropy = {}

for num_neurons in num_neurons_array:

	print('num_neurons: %d'%(num_neurons))

	# Create the model
	x = tf.placeholder(tf.float32, [batch_size, NUM_FEATURES])
	y_ = tf.placeholder(tf.float32, [batch_size, NUM_CLASSES])

	# Build the graph for the deep net

	#input
	#hidden perceptron
	#output softmax

	#init weights
	w1 = tf.Variable(tf.truncated_normal([NUM_FEATURES, num_neurons], stddev=1.0/math.sqrt(float(NUM_FEATURES))), name='w1')
	w2 = tf.Variable(tf.truncated_normal([num_neurons, NUM_CLASSES], stddev=1.0/math.sqrt(float(NUM_FEATURES))), name='w2')
	b1  = tf.Variable(tf.zeros([num_neurons]), name='b1')
	b2 = tf.Variable(tf.zeros([NUM_CLASSES]), name='b2')

	#hidden layer
	z  = tf.matmul(x, w1) + b1
	h = tf.sigmoid(z)

	#output layer
	logits = tf.matmul(h,w2)+b2

	cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
	#mean cross entropy
	loss = tf.reduce_mean(cross_entropy)
	entropy_loss = tf.reduce_mean(tf.cast(loss,tf.float32))
	
	#l2
	regularizer = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)
	loss = tf.reduce_mean(loss + beta * regularizer)

	# Create the gradient descent optimizer with the given learning rate.
	#train = tf.train.batch(batch_size,allow_smaller_final_batch=True)
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	train_op = optimizer.minimize(loss)


	correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)
	wrong_prediction = tf.cast(tf.not_equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)

	errors = tf.reduce_sum(wrong_prediction)
	accuracy = tf.reduce_mean(correct_prediction)


	#create dictionary listing
	batch_train_acc[num_neurons] = []
	batch_train_err[num_neurons] = []
	batch_train_cross_entropy[num_neurons] = []


	start_train_time = time.time()
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		test_acc = []
		
		for i in range(epochs):
			if i == epochs-1:
				single_epoch_time_start = time.time()
			temp_acc = []
			temp_err = []
			temp_ent = []
			for j in range(0,trainX.shape[0],batch_size):
			
				if(j+batch_size < trainX.shape[0]):			
					batch_start,batch_end = j,j+batch_size
				else:
					batch_start,batch_end = trainX.shape[0]-batch_size,trainX.shape[0]
					
				train_op.run(feed_dict={x: trainX[batch_start:batch_end], y_: trainY[batch_start:batch_end]})
				
				j_err = errors.eval(feed_dict={x: trainX[batch_start:batch_end], y_: trainY[batch_start:batch_end]})
				j_acc = accuracy.eval(feed_dict={x: trainX[batch_start:batch_end], y_: trainY[batch_start:batch_end]})
				j_ent = entropy_loss.eval(feed_dict={x: trainX[batch_start:batch_end], y_: trainY[batch_start:batch_end]})
				temp_err.append(j_err)
				temp_acc.append(j_acc)
				temp_ent.append(j_ent)
		
			batch_train_acc[num_neurons].append(np.average(temp_acc))
			batch_train_err[num_neurons].append(np.sum(temp_err))
			batch_train_cross_entropy[num_neurons].append(np.average(temp_ent))

			if (i % 100 == 0) or (i == (epochs-1)):
				if i == (epochs-1):
					single_epoch_time_end = time.time()
					single_epoch_time = single_epoch_time_end - single_epoch_time_start
					batch_single_epoch.append(single_epoch_time)
				print('iter %d: accuracy %g, error %g, entropy %g'%(i, batch_train_acc[num_neurons][i],batch_train_err[num_neurons][i],batch_train_cross_entropy[num_neurons][i]))
				
		#stop timer
		end_train_time = time.time()
		batch_time = end_train_time - start_train_time
		batch_time_set.append(batch_time)
		#test set
		temp_acc = []
		temp_err = []
		for j in range(0,testX.shape[0],batch_size):
			
			if(j+batch_size < testX.shape[0]):			
				batch_start,batch_end = j,j+batch_size
			else:
				batch_start,batch_end = testX.shape[0]-batch_size,testX.shape[0]
				
			j_err = errors.eval(feed_dict={x: testX[batch_start:batch_end], y_: testY[batch_start:batch_end]})
			j_acc = accuracy.eval(feed_dict={x: testX[batch_start:batch_end], y_: testY[batch_start:batch_end]})
			temp_acc.append(j_acc)
			temp_err.append(j_err)
		
		
		test_acc = np.average(temp_acc)
		test_err = np.sum(temp_err)
		batch_test_acc.append(test_acc)
		batch_test_err.append(test_err)
		print('test accuracy: accuracy %g, error %g'%(test_acc,test_err))


		
#process time array
for i in range(len(batch_time_set)):
	batch_time_set[i] = batch_time_set[i]/60

print('\n')
for i in range(len(num_neurons_array)):
	print('test_num_neurons_%d: %g, err: %g'%(num_neurons_array[i],batch_test_acc[i],batch_test_err[i]))
	
	
# plot learning curves
#BATCH_SIZE/TRAIN_ACC
plt.figure(1)
for i in batch_train_acc.keys():
	plt.plot(range(epochs), batch_train_acc[i], label='num_neurons = {}'.format(i))
plt.xlabel('iterations')
plt.legend()
plt.ylabel('train accuracy')
plt.savefig('neurons_1.png')

#BATCH_SIZE/TEST_ACC
plt.figure(2)
plt.plot(range(len(num_neurons_array)),batch_test_acc, label='num_neurons = {}'.format(i))
plt.xlabel('num_neurons')
plt.xticks(range(len(num_neurons_array)), num_neurons_array)
plt.ylabel('test accuracy')
plt.savefig('neurons_2.png')

#BATCH_SIZE/TRAIN_TIME
plt.figure(3)
plt.plot(range(len(num_neurons_array)), batch_time_set)
plt.xticks(range(len(num_neurons_array)), num_neurons_array)
plt.xlabel('num_neurons.')
plt.ylabel('time to train (min)')
plt.savefig('neurons_3.png')

#BATCH_SIZE/TRAIN_TIME_SINGLE_EPOCH
plt.figure(4)
plt.plot(range(len(num_neurons_array)), batch_single_epoch)
plt.xticks(range(len(num_neurons_array)), num_neurons_array)
plt.xlabel('batch size')
plt.ylabel('time to train (s)')
plt.savefig('neurons_4.png')


#BATCH_SIZE/TRAIN_ERR
plt.figure(5)
for i in batch_train_err.keys():
	plt.plot(range(epochs), batch_train_err[i], label='num_neurons = {}'.format(i))
plt.xlabel(str(epochs) + ' iterations')
plt.legend()
plt.ylabel('train errors')
plt.savefig('neurons_5.png')

#BATCH_SIZE/TEST_ERR
plt.figure(6)
plt.plot(range(len(num_neurons_array)),batch_test_err, label='num_neurons = {}'.format(i))
plt.xlabel('num_neurons.')
plt.xticks(range(len(num_neurons_array)), num_neurons_array)
plt.ylabel('test error')
plt.savefig('neurons_6.png')

# CROSS_ENTROPY
plt.figure(7)
for i in batch_train_cross_entropy.keys():
	plt.plot(range(epochs),batch_train_cross_entropy[i],label='num_neurons = {}'.format(i))
plt.xlabel('iterations.')
plt.ylabel('cross entropy')
plt.legend()
plt.savefig('neurons_7.png')
plt.show()