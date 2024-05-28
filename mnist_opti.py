import keras_tuner
import keras
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(60000,1,28,28,1)
x_test = x_test.reshape(10000,1,28,28,1)




def build_model(hp):
  model = keras.Sequential()
  model.add(keras.layers.ConvLSTM2D(12,(9,4),dropout=0.2,return_sequences=True,activation='relu'))
  model.add(keras.layers.ConvLSTM2D(12,(4,9),dropout=0.2,return_sequences=True,activation='relu'))
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(15, activation='silu'))
  model.add(keras.layers.Dense(10, activation=hp.Choice('units', ['softmax','sigmoid','silu'])))
  model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
  return model

tuner = keras_tuner.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=8)
tuner.search(x_train, y_train, epochs=8, validation_data=(x_test[:5000], y_test[:5000]))
best_model = tuner.get_best_models()[0]

best_model.evaluate(x_test[5000:], y_test[5000:])