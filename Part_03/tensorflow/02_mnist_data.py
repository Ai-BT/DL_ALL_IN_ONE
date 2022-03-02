
# %%

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets

mnist = datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x.shape
# %%

image = train_x[0]
image.shape

plt.imshow(image, 'gray')
plt.show()
# %%
