import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt

model = load_model('mnistModel/mnistCNNModel.h5')

img_path = '7.png'
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(28, 28), color_mode="grayscale")
plt.imshow(img, cmap='gray')
plt.show()

x = image.img_to_array(img)
x = 255 - x
x /= 255
x = np.expand_dims(x, axis=0)

prediction = model.predict(x)
print(prediction)

print("ИИ думает, что это цифра: ", np.argmax(prediction))