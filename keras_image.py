import numpy as np

import tensorflow as tf
import tensorflow.contrib.keras as keras

image = keras.preprocessing.image
inception_v3 = keras.applications.inception_v3
preprocess_input = keras.applications.inception_v3.preprocess_input
decode_predictions = keras.applications.inception_v3.decode_predictions

model = inception_v3.InceptionV3(weights='imagenet')

img_path = './elephant.jpg'
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = inception_v3.preprocess_input(x)

preds = model.predict(x)

print('Predicted:', inception_v3.decode_predictions(preds, top=3)[0])
