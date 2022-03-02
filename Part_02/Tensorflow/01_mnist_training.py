# %%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import datasets

# Data Load
(train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()
train_x, test_x = train_x / 255.0, test_x / 255.0
# %%

# Model 생성
model = tf.keras.models.Sequential([
    layers.Flatten(input_shape=(28,28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer = 'adam',
            loss = 'sparse_categorical_crossentropy',
            metrics=['accuracy'])

# %%

# Training / Ecaluation
model.fit(train_x, train_y, epochs=10)

model.evaluate(test_x, test_y)

# %%
