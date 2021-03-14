import sys
sys.path.insert(1, 'panotti-master/panotti')

from datautils import *
from models import *
import tensorflow as tf
import numpy as np
import torch

# for any direction q and step size e, x + eq | x - eq is likely to decrease probability of correct class
# repeatedly pick random directions q and add or subtract them
# to minimize queries to h(.) - we always try to add eq

# MODEL:
# takes input, target audio label pair (x,y), set of orthonomal candidate vectors Q and step-size e > 0
# for simplicity pick q E Q uni-formly at random
# to guarantee maximum query efficiency, ensure that no two directions cancel each other out and diminish progress
# pick q without replacement
# restrict all vectors in Q to be orthonomal

# get the data
X_test, Y_test, paths_test, class_names = build_dataset(path="panotti-master/Preproc/Test/", batch_size=40)

# load model

model, serial_model = models.setup_model(X_test, class_names, weights_file="panotti-master/weights.hdf5", missing_weights_fatal=True)
model.summary()

def get_confidence(model,x):
    prediction = model.predict(np.array([x]))
    return(np.argmax(prediction))


def SimBA_attack(model, x, y, iterations=10000, epsilon=0.2):
    # get dimensions of x and flatten
    dimensions = (tf.reshape(x,[1,-1])).get_shape()
    permutations = tf.convert_to_tensor(np.random.permutation(dimensions[1]))

    probability = get_confidence(model,x)

    for i in range(iterations):
        perturbation = tf.zeros(dimensions)
        j = np.array(permutations[i])

        perturbation = np.array(perturbation[0])
        perturbation[0] = epsilon
        perturbation = tf.convert_to_tensor(perturbation)
        print(probability)
        left_prob = get_confidence(model,x-perturbation)
        exit()
    '''
    n_dims = x.view(1,-1).size(1)
    perms = torch.randperm(n_dims)
    '''
    # generate random permutation for dimensions between -1 and 1 for dimensions of the tensor

    # get probability of classification

    # iterate num_iters
        # x + eq

        # x - eq

SimBA_attack(model,X_test[0],Y_test[0])