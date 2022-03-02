# %%

import numpy as np
import tensorflow as tf


# 1. Tensor 생성

# list -> Tensor 변환
tf.constant([1,2,3])

# tuple -> Tensor 변환
tf.constant(((1, 2, 3), (1, 2, 3)))

# %%

# Array -> Tensor 변환
# 거의 array를 tensor로 변환한다.
arr = np.array([1, 2, 3])
print(arr)

tensor = tf.constant(arr)
print(tensor.shape)
print(tensor)

# %%

# 2. 난수 생성

np_ran = np.random.randn(9)
tf_ran_normal = tf.random.normal([3, 3]) # 표준정규분포에 따른 난수
tf_ran_uni = tf.random.uniform([4, 4]) # 평균값이 고려가 되지 않는 난수

print('np_ran = ',np_ran)
print('tf_ran_normal = ',tf_ran_normal)
print('tf_ran_uni = ',tf_ran_uni)


# %%
