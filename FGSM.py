import math
import sys
sys.path.insert(1, 'panotti-master/panotti')

from datautils import *
from models import *
import tensorflow as tf
import numpy as np
import librosa
import soundfile as sf
import skimage.io
from tqdm import tqdm

# get the data
X_test, Y_test, paths_test, class_names = build_dataset(path="panotti-master/Preproc/Test/", batch_size=40)

# load model

model, serial_model = models.setup_model(X_test, class_names, weights_file="panotti-master/weights.hdf5", missing_weights_fatal=True)
model.summary()

cat_loss = tf.keras.losses.CategoricalCrossentropy()

def calculate_snr(audio,perturbation):
    audio = librosa.feature.inverse.mel_to_audio(np.array(audio.squeeze()), sr=44100)
    perturbation = librosa.feature.inverse.mel_to_audio(np.array(perturbation).squeeze(), sr=44100)
    audio_rms = math.sqrt(np.mean(audio**2))
    perturbation_rms = math.sqrt(np.mean(perturbation**2))
    snr = 10 * math.log10(audio_rms/perturbation_rms)
    print(snr)



def fgsm(audio, label, model, epsilon = 0.1):
    tensor_audio = tf.convert_to_tensor(audio)

    with tf.GradientTape() as tape:
        tape.watch(tensor_audio)
        prediction = model(tensor_audio)
        label = tf.one_hot(decode_class(label, class_names), prediction.shape[-1])
        label = tf.reshape(label, (1, prediction.shape[-1]))

        loss = cat_loss(label,prediction)

    gradient = tape.gradient(loss,tensor_audio)
    signed_grad = tf.sign(gradient)
    calculate_snr(audio,(epsilon * signed_grad))
    return (audio + (epsilon * signed_grad))

num_predictions = X_test.shape[0]
predictions = []


for i in tqdm(range(0,num_predictions)):
    predictions.append(model(fgsm(X_test[i:i+1,:,:,:],Y_test[i],model,epsilon=0.1)).numpy())


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def spectrogram_to_image(mels):
    mels = np.log(mels.squeeze() + 1e-9)

    img = scale_minmax(mels, 0, 255).astype(np.uint8)
    img = np.flip(mels, axis=0)

    return img

def test_accuracy(scores,actual,class_names,predictions):
    total = 0
    correct = 0

    for i in range(predictions):
        prediction = decode_class(scores[i],class_names)
        real = decode_class(actual[i],class_names)
        total += 1
        if (prediction == real):
            correct += 1

    print("accuracy is "+ str((correct/total)*100) + "%")

'''
test_accuracy(predictions,Y_test,class_names,num_predictions)
test_accuracy(model.predict(X_test,batch_size = 40),Y_test,class_names,num_predictions)

test = librosa.feature.inverse.mel_to_audio(np.array(X_test[0].squeeze()), sr=44100)
sf.write('fgsm/original.wav', test, 44100)
skimage.io.imsave('fgsm/original_spectrogram.png', spectrogram_to_image(X_test[1]))

test = np.array(fgsm(X_test[0:1,:,:,:],Y_test[0],model,epsilon=0.1))
test = librosa.feature.inverse.mel_to_audio(test.squeeze(), sr=44100)
sf.write('fgsm/adv.wav', test, 44100)
skimage.io.imsave('fgsm/spectrogram.png', spectrogram_to_image(np.array(fgsm(X_test[0:1,:,:,:],Y_test[0],model,epsilon=0.1))))
'''