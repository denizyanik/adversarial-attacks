import sys
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

cat_loss = tf.keras.losses.CategoricalCrossentropy()

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
    print(audio)
    print(audio + (epsilon * signed_grad))
    exit()
    return (model(audio + (epsilon * signed_grad)))

num_predictions = X_test.shape[0]
predictions = []


for i in range(0,num_predictions):
    predictions.append(fgsm(X_test[i:i+1,:,:,:],Y_test[i],model,epsilon=0.1).numpy())


def test_accuracy(scores,actual,class_names,predictions):
    total = 0
    correct = 0
    print(actual.shape[0])
    for i in range(predictions):
        prediction = decode_class(scores[i],class_names)
        real = decode_class(actual[i],class_names)
        total += 1
        if (prediction == real):
            correct += 1

    print("accuracy is "+ str((correct/total)*100) + "%")

test_accuracy(predictions,Y_test,class_names,num_predictions)
test_accuracy(model.predict(X_test,batch_size = 40),Y_test,class_names,num_predictions)

