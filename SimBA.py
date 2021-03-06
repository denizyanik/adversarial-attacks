import math
import sys

import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

sys.path.insert(1, 'panotti-master/panotti')

from datautils import *
from models import *
import tensorflow as tf
import numpy as np
import torch
import librosa
import soundfile as sf
import skimage.io
from tqdm import tqdm
import matplotlib.pyplot as plt
import librosa.display

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
snr = 0

def calculate_snr(audio,perturbation):
    audio = librosa.feature.inverse.mel_to_audio(np.array(audio.squeeze()), sr=44100)
    perturbation = librosa.feature.inverse.mel_to_audio(np.array(perturbation).squeeze(), sr=44100)
    audio_rms = np.mean(audio)
    perturbation_rms = np.mean(perturbation)
    snr = 10 * math.log10((audio_rms/perturbation_rms)*(audio_rms/perturbation_rms))
    print(snr)
    return snr

def get_confidence(model,x,y):
    prediction = model.predict(x)
    pos = decode_class(y,class_names)
    return(prediction[0][pos])
    #return(np.amax(prediction))


def SimBA_attack(model, x, y, iterations=1000, epsilon=0.1):
    global snr
    # get dimensions of x and flatten
    dimensions = (tf.reshape(x,[1,-1])).get_shape()
    permutations = tf.convert_to_tensor(np.random.permutation(dimensions[1]))


    probability = get_confidence(model,x,y)
    snr_pert = tf.zeros(tf.shape(x))
    snr_audio = np.copy(x)
    for i in range(iterations):
        perturbation = tf.zeros(dimensions)
        j = np.array(permutations[i])
        perturbation = np.array(perturbation[0])
        perturbation[j] = epsilon
        perturbation = tf.convert_to_tensor(perturbation)

        left_prob = get_confidence(model,x-tf.reshape(perturbation,tf.shape(x)),y)

        if False != (left_prob < probability):
            x = x-tf.reshape(perturbation,tf.shape(x))
            probability = left_prob
            snr_pert += tf.reshape(perturbation,tf.shape(x))
        else:
            right_prob = get_confidence(model,x+tf.reshape(perturbation,tf.shape(x)),y)
            if False != (right_prob < probability):
                x = x+tf.reshape(perturbation,tf.shape(x))
                probability = right_prob
                snr_pert += tf.reshape(perturbation,tf.shape(x))

    snr += calculate_snr(snr_audio,snr_pert)
    return x


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

# test accuracy

total = 0
correct = 0
r_correct = 0
'''
for i in tqdm(range(0,X_test.shape[0])):

    r_check = model.predict(X_test[i:i + 1, :, :, :])
    r_prediction = decode_class(r_check, class_names)

    real = decode_class(Y_test[i], class_names)

    adversarial_example = SimBA_attack(model,X_test[i:i+1,:,:,:],Y_test[i])
    adversarial_example = model.predict(adversarial_example)


    prediction = decode_class(adversarial_example,class_names)

    total += 1

    if (prediction == real):
        correct += 1
    if (r_prediction == real):
        r_correct += 1

    print("current adversarial accuracy is "+ str((correct/total)*100) + "%")
    print("current accuracy is "+ str((r_correct/total)*100) + "%")
    print(snr/total)
'''

audio = load_melgram("C:/Users/deniz/PycharmProjects/adversarial-attacks/panotti-master/Preproc/Test/Chorus/P64-43110-3311-46225.wav.npz")
x = np.zeros([1, 96, 173, 1])
use_len = min(x.shape[2],audio.shape[2])
audio = np.float32(audio[:,:,0:use_len])

y = 0
for Y in Y_test:
    if np.argmax(Y) ==0:
        y = Y

test = librosa.feature.inverse.mel_to_audio(np.array(audio[0].squeeze()), sr=44100)
sf.write('simba/original.wav', test, 44100)
#imageio.imwrite('simba/original_spectrogram.png', spectrogram_to_image(X_test[0]))
fig = spec_to_image(test)
fig.savefig('simba/original.png')

copy1 = np.copy(audio)
copy2 = np.copy(audio)
copy3 = np.copy(audio)
copy4 = np.copy(audio)
copy5 = np.copy(audio)


# epsilon 0.5
check = (SimBA_attack(model,copy4,y,epsilon=0.2))
print(np.linalg.norm(np.array(check - copy5)))

c = decode_class(model.predict(check),class_names)

test = librosa.feature.inverse.mel_to_audio(np.array(check).squeeze(), sr=44100)
sf.write('simba/simba-0.5.wav', test, 44100)
fig = spec_to_image(test)
fig.savefig('simba/simba-0.5.png')

# epsilon 0.2
check = (SimBA_attack(model,copy1,y,epsilon=0.2))
print(np.linalg.norm(np.array(check - copy5)))

c = decode_class(model.predict(check),class_names)

test = librosa.feature.inverse.mel_to_audio(np.array(check).squeeze(), sr=44100)
sf.write('simba/simba-0.2.wav', test, 44100)
fig = spec_to_image(test)
fig.savefig('simba/simba-0.2.png')

#epsilon 0.1
check = (SimBA_attack(model,audio,y))
print(np.linalg.norm(np.array(check - copy5)))

c = decode_class(model.predict(check),class_names)

test = librosa.feature.inverse.mel_to_audio(np.array(check).squeeze(), sr=44100)
sf.write('simba/simba-0.1.wav', test, 44100)
fig = spec_to_image(test)
fig.savefig('simba/simba-0.1.png')

#epsilon 0.05

check = (SimBA_attack(model,copy2,y,epsilon=0.05))
print(np.linalg.norm(np.array(check - copy5)))

c = decode_class(model.predict(check),class_names)

test = librosa.feature.inverse.mel_to_audio(np.array(check).squeeze(), sr=44100)
sf.write('simba/simba-0.05.wav', test, 44100)
fig = spec_to_image(test)
fig.savefig('simba/simba-0.05.png')

# epsilon 0.01

check = (SimBA_attack(model,copy3,y,epsilon=0.01))
print(np.linalg.norm(np.array(check - copy5)))

c = decode_class(model.predict(check),class_names)

test = librosa.feature.inverse.mel_to_audio(np.array(check).squeeze(), sr=44100)
sf.write('simba/simba-0.01.wav', test, 44100)
fig = spec_to_image(test)
fig.savefig('simba/simba-0.01.png')
