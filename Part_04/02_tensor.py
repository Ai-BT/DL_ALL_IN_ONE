# %%
import tensorflow as tf
import numpy as np
from torch import tensor

# 1. 기본적인 텐서형태 만들기
hello = tf.constant([3,3], dtype=tf.float32)
print(hello)

x = tf.constant([[1.0, 2.0],
                [3.0, 4.0]])
print(x)

# %%

# 2. 아래와 같이 numpy, list 도 tensor로 변환 가능
x_np = np.array([[1.0, 2.0],
                [3.0, 4.0]])

x_list = ([[1.0, 2.0],
        [3.0, 4.0]])

print(type(x_np))
print(type(x_list))

x_np = tf.convert_to_tensor(x_np)
x_list = tf.convert_to_tensor(x_list)

print(type(x_np))
print(type(x_list))

# %%

# 3. Varialbe 변할 수 있는 상태를 저장하는데 사용되는 특별한 텐서 입니다.
#   딥러닝에서 학습해야하는 가중치(weight, bias)들을 variable로 생성

tensor = tf.ones((3,4))
print(tensor)

# variable로 선언
variable = tf.Variable(tensor)
print(variable)

# assign 으로 해당 위치 변경
variable[0,0].assign(2)
print(variable)


# %%

# 4. concat 새로운 행렬 생성

z = tf.range(1,11)
z = tf.reshape(z, (2,5))
print(z)

concat = tf.concat([z, z], axis=0)
print(concat)

concat = tf.concat([z, z], axis=1)
print(concat)


# %%
