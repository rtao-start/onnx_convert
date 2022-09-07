import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np

#model = keras.models.load_model('fashion_model.h5')
model = keras.models.load_model('sm/')

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print('test_images:', test_images.shape)

predictions = model.predict(test_images)

np.save('./result', predictions)

print(np.argmax(predictions[:10], 1))