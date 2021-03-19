import math
import sys

from tqdm import tqdm

sys.path.insert(1, 'panotti-master/panotti')

from datautils import *
from models import *
import tensorflow as tf
import numpy as np


# get the data
X_test, Y_test, paths_test, class_names = build_dataset(path="panotti-master/Preproc/Test/", batch_size=40)

# load model

model, serial_model = models.setup_model(X_test, class_names, weights_file="panotti-master/weights.hdf5", missing_weights_fatal=True)
model.summary()

def zoo_adam_attack(x,y):
    #ADAM variables
    mt = 0
    vt = 0
    beta1 = 0.9
    beta2 = 0.999
    learn_rate = 1e-1
    eps = 1e-8
    h = 0.0001
    batch = 1
    iterations = 1000

    perturbation = np.zeros([1,96,173,1])

    # used to evaluate gradient
    x1 = x
    x2 = x

    for i in range(1,iterations+1):

        size = x[0].size

        # choose a random coordinate and compute partial gradient
        coordinate = np.random.choice(size,1,replace=True)[0]

        x1.reshape(-1)[coordinate] += h
        x2.reshape(-1)[coordinate] -= h

        gradient = np.zeros(x.reshape(-1).shape)
        x1_loss = get_loss(x1,y)
        x2_loss = get_loss(x2,y)
        gradient[coordinate] = (x1_loss - x2_loss / (2*h))

        mt = beta1 * mt + (1-beta1) * gradient
        vt = beta2 * vt + (1-beta2) * np.square(gradient)
        corr = (math.sqrt(1 - beta2 ** i)) / (1 - beta1 ** i)

        m = perturbation.reshape(-1)
        m -= learn_rate * corr * (mt / (np.sqrt(vt) + eps))
        perturbation = m.reshape(perturbation.shape)

        x += perturbation

    return x


def get_loss(x,y):

    output = model.predict(x)

    true_lab = tf.one_hot(decode_class(y, class_names), output.shape[-1])
    true_lab = tf.reshape(true_lab, (1, output.shape[-1]))

    # gets Z value of real class
    real = tf.reduce_sum((true_lab) * output, 1)

    # gets most likely other class
    # minus 10000 for real class as Z values can be negative and we don't want to choose real class as max
    other = tf.reduce_max((1-true_lab)*output - (true_lab*10000),1)

    # optimize for making real class least likely as this will be an untargeted attack
    loss = tf.maximum(0.0, tf.math.log(real + 1e-30) - tf.math.log(other + 1e-30))

    return loss


total = 0
correct = 0
r_correct = 0

# test accuracy
for i in tqdm(range(0,X_test.shape[0])):
    r_check = model.predict(X_test[i:i + 1, :, :, :])
    r_prediction = decode_class(r_check, class_names)

    adversarial_example = zoo_adam_attack(X_test[i:i+1, :, :, :], Y_test[i])
    adversarial_example = model.predict(adversarial_example)

    real = decode_class(Y_test[i],class_names)
    prediction = decode_class(adversarial_example,class_names)

    total += 1

    print(real)
    print(prediction)
    print((r_prediction))

    if (prediction == real):
        correct += 1
    if (r_prediction == real):
        r_correct += 1

    print("current adversarial accuracy is "+ str((correct/total)*100) + "%")
    print("current accuracy is "+ str((r_correct/total)*100) + "%")



