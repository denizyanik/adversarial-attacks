import math
import sys

from matplotlib.backends.backend_template import FigureCanvas

sys.path.insert(1, 'panotti-master/panotti')

from datautils import *
from models import *
import tensorflow as tf
import numpy as np
import librosa
import soundfile as sf
import skimage.io
from tqdm import tqdm
import matplotlib.pyplot as plt
import librosa.display
from keras.utils.vis_utils import plot_model
from keras_visualizer import visualizer
from numpy import load

# get the data
X_test, Y_test, paths_test, class_names = build_dataset(path="panotti-master/Preproc/Test/", batch_size=40)

# load model

model, serial_model = models.setup_model(X_test, class_names, weights_file="panotti-master/weights.hdf5", missing_weights_fatal=True)
model.summary()
#tf.keras.utils.plot_model(model=model,to_file="img.png",show_shapes=True,show_layer_names=True)

cat_loss = tf.keras.losses.CategoricalCrossentropy()

def calculate_snr(audio,perturbation):
    audio = librosa.feature.inverse.mel_to_audio(np.array(audio.squeeze()), sr=44100)
    perturbation = librosa.feature.inverse.mel_to_audio(np.array(perturbation).squeeze(), sr=44100)
    audio_rms = np.mean(audio)
    perturbation_rms = np.mean(perturbation)
    snr = 10 * math.log10((audio_rms / perturbation_rms) * (audio_rms / perturbation_rms))
    return snr

snr = 0

def fgsm(audio, label, model, epsilon = 0.1):
    global snr
    tensor_audio = tf.convert_to_tensor(audio)
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    with tf.GradientTape() as tape:
        tape.watch(tensor_audio)
        prediction = model(tensor_audio)
        label = tf.one_hot(decode_class(label, class_names), prediction.shape[-1])
        label = tf.reshape(label, (1, prediction.shape[-1]))

        loss = loss_object(label,prediction)

    gradient = tape.gradient(loss,tensor_audio)
    signed_grad = tf.sign(gradient)
    snr += calculate_snr(audio,(epsilon * signed_grad))
    return (audio + (epsilon * signed_grad))

num_predictions = X_test.shape[0]
predictions = []
'''
for i in tqdm(range(0,num_predictions)):
    predictions.append(model(fgsm(X_test[i:i+1,:,:,:],Y_test[i],model,epsilon=0.1)).numpy())
'''
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
    print(snr/total)


#test_accuracy(predictions,Y_test,class_names,num_predictions)
#test_accuracy(model.predict(X_test,batch_size = 40),Y_test,class_names,num_predictions)


audio = load_melgram("C:/Users/deniz/PycharmProjects/adversarial-attacks/panotti-master/Preproc/Test/Chorus/P64-43110-3311-46225.wav.npz")
x = np.zeros([1, 96, 173, 1])
use_len = min(x.shape[2],audio.shape[2])
audio = np.float32(audio[:,:,0:use_len])

test = librosa.feature.inverse.mel_to_audio(np.array(audio[0].squeeze()), sr=44100)

sf.write('fgsm/original.wav', test, 44100)
fig = spec_to_image(test)
fig.savefig('fgsm/original.png')

y = 0
for Y in Y_test:
    if np.argmax(Y) ==0:
        y = Y

copy1 = np.copy(audio)
copy2 = np.copy(audio)
copy3 = np.copy(audio)
copy4 = np.copy(audio)
copy5 = np.copy(audio)

#epsilon = 0.5

test = np.array(fgsm(copy4,y,model,epsilon=0.5))
print(np.linalg.norm(np.array(test - copy5)))

test = librosa.feature.inverse.mel_to_audio(test.squeeze(), sr=44100)
sf.write('fgsm/adv-0.5.wav', test, 44100)
fig = spec_to_image(test)
fig.savefig('fgsm/spectrogram-0.5.png')


#epsilon = 0.2
test = np.array(fgsm(copy1,y,model,epsilon=0.2))
print(np.linalg.norm(np.array(test - copy5)))
test = librosa.feature.inverse.mel_to_audio(test.squeeze(), sr=44100)
sf.write('fgsm/adv-0.2.wav', test, 44100)
fig = spec_to_image(test)
fig.savefig('fgsm/spectrogram-0.2.png')

#epsilon = 0.1
test = np.array(fgsm(audio,y,model,epsilon=0.1))
print(np.linalg.norm(np.array(test - copy5)))
test = librosa.feature.inverse.mel_to_audio(test.squeeze(), sr=44100)
sf.write('fgsm/adv-0.1.wav', test, 44100)
fig = spec_to_image(test)
fig.savefig('fgsm/spectrogram-0.1.png')

#epsilon = 0.05
test = np.array(fgsm(copy2,y,model,epsilon=0.05))
print(np.linalg.norm(np.array(test - copy5)))
test = librosa.feature.inverse.mel_to_audio(test.squeeze(), sr=44100)
sf.write('fgsm/adv-0.05.wav', test, 44100)
fig = spec_to_image(test)
fig.savefig('fgsm/spectrogram-0.05.png')

#epsilon = 0.01
test = np.array(fgsm(copy3,y,model,epsilon=0.01))
print(np.linalg.norm(np.array(test - copy5)))
test = librosa.feature.inverse.mel_to_audio(test.squeeze(), sr=44100)
sf.write('fgsm/adv-0.01.wav', test, 44100)
fig = spec_to_image(test)
fig.savefig('fgsm/spectrogram-0.01.png')
