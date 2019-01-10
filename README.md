# nn_project
PART A – CLASSIFICATION PROBLEM
Most of the experiments are conducted using a feedforward neural network with hidden perceptron layer(s) and a output softmax layer, with L2 regularisation implemented. For weight training, minibatch gradient descent is used as per the question requirement.
Sequential mini batch training is used, where if there is a final small batch, we sample the 32 samples of the dataset again. Thus, one epoch may sample more than 4435 rows of data.

PART B : REGRESSION PROBLEM

Given the California Housing Price Dataset which contains attributes of housing complexes in California such as location, size of house, etc, together with their corresponding prices. We are supposed to build a model that estimate the pricing given its attributes.
