# %%

import tensorflow as tf
import os 
import matplotlib.pyplot as plt
from tensorflow.keras import datasets

(train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()
image = train_x[0]

# 이미지 shape 확인
image.shape

plt.imshow(image, 'gray')
plt.show()

# %%


# %%
