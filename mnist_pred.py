import tensorflow as tf
import keras

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(60000,1,28,28,1)
x_test = x_test.reshape(10000,1,28,28,1)



model = keras.Sequential()
model.add(keras.layers.ConvLSTM2D(12,(9,4),dropout=0.33 ,return_sequences=True,activation='relu'))
model.add(keras.layers.ConvLSTM2D(12,(4,9),dropout=0.33,return_sequences=True,activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(15, activation='silu'))
model.add(keras.layers.Dense(10, 'sigmoid'))

model.compile(optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])

model.fit(x_train,y_train, epochs = 5  , validation_data=(x_test[:5000], y_test[:5000]))
model.evaluate(x_test[5000:], y_test[5000:])

#silu 