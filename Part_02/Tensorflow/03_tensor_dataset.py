# %%
import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets
import matplotlib.pyplot as plt
from keras.utils import np_utils

# 1. Data_Load
mnist = datasets.fashion_mnist
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.shape, train_labels.shape) # traing shape 확인
print(test_images.shape, test_labels.shape) # test shape 확인

print(type(train_images)) # 타입 확인

# %%

# 2. Data class 별 확인
# trainin set의 각 class 별 image 수 확인
unique, counts = np.unique(train_labels, axis=-1, return_counts=True)
train_set = dict(zip(unique, counts))
print(train_set)

# test set의 각 class 별 image 수 확인
unique, counts = np.unique(test_labels, axis=-1, return_counts=True)
test_set = dict(zip(unique, counts))
print(test_set)

# %%

# 3. 시각화

plt.figure(figsize=(8,8))

for i in range(9):
        plt.subplot(3,3, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap='gray')
        plt.title(class_names[i])

plt.show()

# %%

# 4. 데이터 전처리
# image를 0 ~ 1 사이 값으로 변경 / 255
train_images = train_images.astype(np.float32) / 255.
test_images = test_images.astype(np.float32) / 255.

# one-hot encoding
# 정답의 경우 10개의 output을 내보내도록 할 것이다.
# 10개의 클래스 확률로 나뉘는 것
train_labels = np_utils.to_categorical(train_labels, 10)
test_labels = np_utils.to_categorical(test_labels, 10)


# %%

# 5. Dataset 만들기
# numpy array 를 from_tensor_slices로 변경
# 셔플을 하는 이유는 똑같은 데이터를 학습 할 수 있기 때문에 방지
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(buffer_size=100000).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(64) # 결과만 확인하기 때문에 셔플을 하지 않음


# %%

# 각 행 마다 해당 이미지가 들어가 있음 (64 배치)
# 레이블은 정답을 가리키는 놈 
for images, labels in train_dataset:
        print(labels)
        break

# %%

# 이미지와 정답을 표시하는지 확인
imgs, lbs = next(iter(train_dataset))
print(f"Feature batch shape:  {imgs.shape}")
print(f"Lable batch shape:  {lbs.shape}")

img = imgs[0]
lb = lbs[0]
plt.imshow(img, cmap='gray')
plt.show()
print(f"Label: {lb}")


# %%

