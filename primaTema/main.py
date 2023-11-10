import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10)
model.save('recunoastereCifre.model')

'''
model = tf.keras.models.load_model('recunoastereCifre.model')
'''

erori, precizie = model.evaluate(x_test, y_test)
print(erori)
print(precizie)

numarImagine = 0
while os.path.isfile(f"cifreTest/cifra{numarImagine}.png"):
    try:
        imagine = cv2.imread(f"cifreTest/cifra{numarImagine}.png")[:, :, 0]
        imagine = np.invert(np.array([imagine]))
        predictie = model.predict(imagine)
        print(f"Cifra detectata in imaginea actuala este: {np.argmax(predictie)}!")
        plt.imshow(imagine[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print(f"Ceva nu a mers bine!")
    finally:
        numarImagine += 1

