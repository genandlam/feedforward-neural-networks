#
# Project 1, starter code part a
#
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
batch_size = 32
num_neurons = 10
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
accuracy = tf.reduce_mean(correct_prediction)
error = tf.reduce_sum(wrong_prediction)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	train_acc = []
	train_err = []
	test_acc = []
	test_err = []
	batch_acc = []
	train_entropy = []
	test_batch_acc = []
	
	for i in range(epochs):	
		temp_acc = []
		temp_err = []
		temp_ent = []
		for j in range(0,trainX.shape[0],batch_size):
		
			if(j+batch_size < trainX.shape[0]):			
				batch_start,batch_end = j,j+batch_size
			else:
				batch_start,batch_end = trainX.shape[0]-batch_size,trainX.shape[0]
				
			train_op.run(feed_dict={x: trainX[batch_start:batch_end], y_: trainY[batch_start:batch_end]})
	
			j_acc = accuracy.eval(feed_dict={x: trainX[batch_start:batch_end], y_: trainY[batch_start:batch_end]})
			j_err = error.eval(feed_dict={x: trainX[batch_start:batch_end], y_: trainY[batch_start:batch_end]})
			j_ent = entropy_loss.eval(feed_dict={x: trainX[batch_start:batch_end], y_: trainY[batch_start:batch_end]})
			temp_acc.append(j_acc)
			temp_err.append(j_err)
			temp_ent.append(j_ent)
			batch_acc.append(j_acc)
			# train_acc.append(accuracy.eval(feed_dict={x: trainX[batch_start:batch_end], y_: trainY[batch_start:batch_end]}))	
		train_entropy.append(np.average(temp_ent))
		train_acc.append(np.average(temp_acc))
		train_err.append(np.sum(temp_err))


		if (i % 100 == 0) or (i== epochs-1):
			print('iter %d: accuracy %g, error %g, entropy %g'%(i, train_acc[i],train_err[i],train_entropy[i]))
			
	#test set
	temp_acc = []
	temp_err = []
	for j in range(0,testX.shape[0],batch_size):
		
		if(j+batch_size < testX.shape[0]):			
			batch_start,batch_end = j,j+batch_size
		else:
			batch_start,batch_end = testX.shape[0]-batch_size,testX.shape[0]

		j_acc = accuracy.eval(feed_dict={x: testX[batch_start:batch_end], y_: testY[batch_start:batch_end]})
		j_error = error.eval(feed_dict={x: testX[batch_start:batch_end], y_: testY[batch_start:batch_end]})
		temp_acc.append(j_acc)
		temp_err.append(j_err)
		test_batch_acc.append(j_acc)
	
	test_acc = np.average(temp_acc)
	test_err = np.sum(temp_err)
	
	
	print('test accuracy: accuracy %g, error %g'%(test_acc,test_err))

	
			
	


# plot learning curves
#TRAIN_ACC
plt.figure(1)
plt.plot(range(epochs), train_acc)
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('train accuracy')
plt.savefig('net_1.png')
#TRAIN_ERR
plt.figure(2)
plt.plot(range(epochs), train_err)
plt.xlabel('iterations')
plt.ylabel('train errors')
plt.savefig('net_2.png')

# plt.figure(2)
# plt.plot(range(epochs*(trainX.shape[0]//batch_size+1)), batch_acc)
# plt.xlabel('batch iterations')
# plt.ylabel('batch accuracy')
# plt.show()

# #TEST_ACC
# plt.figure(3)
# plt.plot(range(len(test_batch_acc)), test_batch_acc)
# plt.xlabel('index')
# plt.ylabel('test accuracy')
# plt.savefig('net_3.png')

# #TEST_ERR
# plt.figure(4)
# plt.plot(range(len(test_err)), test_err)
# plt.xlabel('index')
# plt.ylabel('test error')
# plt.savefig('net_4.png')

#CROSS_ENTROPY
plt.figure(5)
plt.plot(range(epochs),train_entropy)
plt.xlabel('iterations.')
plt.ylabel('cross entropy')
plt.savefig('net_5.png')
plt.show()
