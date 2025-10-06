"""
This is a HelloWorld-type of script to run on the GPU nodes. 
It uses Tensorflow with Keras and is based on this TensorFlow tutorial:
https://www.tensorflow.org/tutorials/keras/classification
"""

# Import TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras

# Some helper libraries
import os
import numpy as np
import matplotlib.pyplot as plt

# Some helper functions
# +++++++++++++++++++++
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# Run some tests
# ++++++++++++++

# get the version of TensorFlow
print("TensorFlow version: {}".format(tf.__version__))

# Check that TensorFlow was build with CUDA to use the gpus
print("Device name: {}".format(tf.test.gpu_device_name()))
print("Build with GPU Support? {}".format(tf.test.is_built_with_gpu_support()))
print("Build with CUDA? {} ".format(tf.test.is_built_with_cuda()))

# Get the data
# ++++++++++++

# Get an example dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Class names for later use
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
               
# Get some information about the data
print("Size of training dataset: {}".format(train_images.shape))
print("Number of labels training dataset: {}".format(len(train_labels)))
print("Size of test dataset: {}".format(test_images.shape))
print("Number of labels test dataset: {}".format(len(test_labels)))

# Convert the data from integer to float
train_images = train_images / 255.0
test_images = test_images / 255.0

# plot the first 25 images of the training Set
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.close('all')

# Set and train the model
# +++++++++++++++++++++++


# Set up the layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy: {}'.format(test_acc))

# Use the model
# +++++++++++++

# grab an image
img_index=10
img = test_images[img_index]
print(img.shape)

# add image to a batch
img = (np.expand_dims(img,0))
print(img.shape)

# to make predictions, add a new layer
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

# predict the label for the image
predictions_img = probability_model.predict(img)

print("Predictions for image {}:".format(img_index))
print(predictions_img[0])
print("Label with highest confidence: {}".format(np.argmax(predictions_img[0])))

# plot it
plt.figure(figsize=(6, 3))
plt.subplot(1,2, 1)
plot_image(img_index, predictions_img[0], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(img_index, predictions_img[0], test_labels)
plt.savefig("./plots/trainingset_prediction_img{}.png".format(img_index),bbox_inches='tight',overwrite=True)