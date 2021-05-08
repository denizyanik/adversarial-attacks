import gc
import math
import sys

import skimage.io
from tqdm import tqdm

sys.path.insert(1, 'panotti-master/panotti')
#from myutils import *
from datautils import *
from models import *
import tensorflow as tf
import numpy as np
import timeit
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# get the data
X_test, Y_test, paths_test, class_names = build_dataset(path="panotti-master/Preproc/Test/", batch_size=40)



# load model

model, serial_model = models.setup_model(X_test, class_names, weights_file="panotti-master/weights.hdf5", missing_weights_fatal=True)
model.summary()
snr = 0
def calculate_snr(audio,perturbation):
    audio = librosa.feature.inverse.mel_to_audio(np.array(audio.squeeze()), sr=44100)
    perturbation = librosa.feature.inverse.mel_to_audio(np.array(perturbation).squeeze(), sr=44100)
    audio_rms = math.sqrt(np.mean(audio**2))
    perturbation_rms = math.sqrt(np.mean(perturbation**2))
    snr = 10 * math.log10((audio_rms/perturbation_rms)*(audio_rms/perturbation_rms))
    return snr

def zoo_adam_attack_batch(xs,ys):
    batch = len(xs)
    mt = [0] * batch
    vt = [0] * batch
    beta1 = 0.9
    beta2 = 0.999
    learn_rate = 1e-1
    eps = 1e-8
    h = 0.0001
    iterations = 1000
    coordinates = [0] * batch

    # variables to calculate gradient
    variables = np.empty([batch*2,96,173,1])
    perturbations = np.zeros([batch,96, 173, 1])

    for i in tqdm(range(1,iterations+1)):

        check = 0
        for z in range(0,batch*2,2):
            var = xs[check:check + 1, :, :, :]
            variables[z] = var
            variables[z + 1] = var
            check += 1

        check = 0
        for j in range(0,batch*2,2):
            size = xs[check:check + 1, :, :, :].size

            # choose a random coordinate and compute partial gradient
            coordinates[check] = np.random.choice(size, 1, replace=True)[0]

            variables[j].reshape(-1)[coordinates[check]] += h
            variables[j+1].reshape(-1)[coordinates[check]] -= h
            check += 1

        # calculate batch losses of all variables to calculate gradient for each sound file
        losses = get_loss_batch(variables,ys,batch)

        # calculate partial gradient
        check = 0
        for j in range(0,batch):

            x = xs[j:j + 1, :, :, :]

            gradient = np.zeros(x.reshape(-1).shape)
            x1_loss = losses[check]
            x2_loss = losses[check+1]

            gradient[coordinates[j]] = (x1_loss - x2_loss / (2 * h))

            mt[j] = beta1 * mt[j] + (1 - beta1) * gradient
            vt[j] = beta2 * vt[j] + (1 - beta2) * np.square(gradient)
            corr = (math.sqrt(1 - beta2 ** i)) / (1 - beta1 ** i)

            m = perturbations[j:j + 1, :, :, :].reshape(-1)
            m -= learn_rate * corr * (mt[j] / (np.sqrt(vt[j]) + eps))
            perturbations[j:j + 1, :, :, :] = m.reshape(perturbations[j:j + 1, :, :, :].shape)

            xs[j:j + 1, :, :, :] += perturbations[j:j + 1, :, :, :]
            check += 2

        del (losses)
        gc.collect()

    return xs




def get_loss_batch(xs,ys,batch):

    outputs = model.predict(xs)
    losses = np.empty(batch*2)

    for i in range(0,batch*2):
        output = outputs[i]
        y = ys[i//2]

        true_lab = tf.one_hot(decode_class(y, class_names), output.shape[-1])
        true_lab = tf.reshape(true_lab, (1, output.shape[-1]))

        # gets Z value of real class
        real = tf.reduce_sum((true_lab) * output, 1)

        # gets most likely other class
        # minus 10000 for real class as Z values can be negative and we don't want to choose real class as max
        other = tf.reduce_max((1 - true_lab) * output - (true_lab * 10000), 1)

        # optimize for making real class least likely as this will be an untargeted attack
        loss = tf.maximum(0.0, tf.math.log(real + 1e-30) - tf.math.log(other + 1e-30))

        losses[i] = loss

    return losses



def zoo_adam_attack(x,y):
    global snr
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
    snr_pert = np.zeros([1, 96, 173, 1])

    perturbation = np.zeros([1, 96, 173, 1])
    copy = np.copy(x)

    for i in range(1,iterations+1):

        x1 = x
        x2 = x

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
        snr_pert += perturbation

    snr += calculate_snr(copy,snr_pert)
    print(snr)
    return (x)


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

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def spec_to_image(spec):
    window_size = 1024
    window = np.hanning(window_size)
    stft = librosa.core.spectrum.stft(spec, n_fft=window_size, hop_length=512, window=window)
    out = 2 * np.abs(stft) / np.sum(window)

    fig = plt.figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), ax=ax, y_axis='log', x_axis='time')
    fig.savefig('spec.png')
    return fig

# test single accuracy

for i in tqdm(range(0,X_test.shape[0])):

    
    test = librosa.feature.inverse.mel_to_audio(X_test[i].squeeze(), sr=44100)
    sf.write('zoo/original'+str(i)+'.wav', test, 44100)
    fig = spec_to_image(test)
    fig.savefig('zoo/original-'+str(i)+'.png')
    r_check = model.predict(X_test[i:i + 1, :, :, :])
    r_prediction = decode_class(r_check, class_names)

    adversarial_example = zoo_adam_attack(X_test[i:i+1, :, :, :], Y_test[i])
    test = librosa.feature.inverse.mel_to_audio(np.array(adversarial_example).squeeze(), sr=44100)
    sf.write('zoo/adv-'+str(i)+'.wav', test, 44100)
    fig = spec_to_image(test)
    fig.savefig('zoo/adv-' + str(i) + '.png')

    adversarial_example = model.predict(adversarial_example)

    real = decode_class(Y_test[i],class_names)
    prediction = decode_class(adversarial_example,class_names)

    total += 1

    if (prediction == real):
        correct += 1
    if (r_prediction == real):
        r_correct += 1

    print("current adversarial accuracy is "+ str((correct/total)*100) + "%")
    print("current accuracy is "+ str((r_correct/total)*100) + "%")
    if i == 3:
        exit()



# test batch accuracy

copy = np.copy(X_test)
adversarials = zoo_adam_attack_batch(X_test,Y_test)
l2_norm = 0
for i in range(0,adversarials.shape[0]):

    adversarial_example = adversarials[i:i+1,:,:,:]
    adversarial_example = model.predict(adversarial_example)

    real = decode_class(Y_test[i], class_names)
    prediction = decode_class(adversarial_example, class_names)

    total += 1

    if (prediction == real):
        correct += 1

    l2_norm += np.linalg.norm(np.array(adversarial_example - copy[i:i + 1, :, :, :]))
    print("current adversarial accuracy is " + str((correct / total) * 100) + "%")
    print(snr/total)
