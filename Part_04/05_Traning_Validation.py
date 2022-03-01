# Traning / Validation
# Keras API 사용

# %%

import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import datasets
from keras.utils import np_utils


# %%

# 1. Data_Load / Data 전처리
mnist = datasets.fashion_mnist
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# print(train_images.shape, train_labels.shape) # traing shape 확인
# print(test_images.shape, test_labels.shape) # test shape 확인
# print(type(train_images)) # 타입 확인

# image를 0~1사이 값으로 만들기 위하여 255로 나누어줌
train_images = train_images.astype(np.float32) / 255.
test_images = test_images.astype(np.float32) / 255.

# one-hot encoding
train_labels = np_utils.to_categorical(train_labels, 10)
test_labels = np_utils.to_categorical(test_labels, 10)

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(
                buffer_size=100000).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(64)


# %%


# 2. Model 만들기
def create_seq_model():
  model = keras.Sequential()
  model.add(keras.layers.Flatten(input_shape=(28, 28)))
  model.add(keras.layers.Dense(128, activation='relu'))
  model.add(keras.layers.Dropout(0.2))
  model.add(keras.layers.Dense(10, activation='softmax'))
  return model

seq_model = create_seq_model()

learning_rate = 0.001
seq_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# %%

# 3. 학습
history = seq_model.fit(train_dataset, epochs=10, validation_data=test_dataset)

# %%
import matplotlib.pyplot as plt

# plot losses
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# %%

## Plot Accuracy
plt.plot(history.history['accuracy'], 'b-', label='acc')
plt.plot(history.history['val_accuracy'], 'r--', label='val_acc')
plt.xlabel('Epoch')
plt.legend()
plt.show()
# %%

# 4. Model 저장하고 불러오기

seq_model.save_weights('seq_model.ckpt')


# %%

# 새로운 모델
seq_model_2 = create_seq_model()
seq_model_2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# %%

# 학습하지 않은 모델 평가
seq_model_2.evaluate(test_dataset)

# %%

seq_model_2.load_weights('seq_model.ckpt')
# %%

seq_model_2.evaluate(test_dataset)
