# %%

# Tensor에서 모델 만드는 방법을 3가지 제공한다
# 1. Keras Sequential API 사용
# 2. Keras Functional API 사용
# 3. Model Class Subclassing 사용

import keras

# 1. Keras Sequential API 사용
def create_seq_model():
        model = keras.Sequential()
        model.add(keras.layers.Flatten(input_shape=(28,28)))
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dropout(0.2)) # 오버피팅 방지 테크닉
        model.add(keras.layers.Dense(10, activation='softmax')) # output 
        return model

seq_model = create_seq_model()
seq_model.summary()

# 앞에 None 은 배치사이즈


# %%

# 2. Keras Functional API 사용
def create_func_mondel():
        inputs = keras.Input(shape=(28,28))
        flatten = keras.layers.Flatten(input_shape=(28,28))(inputs)
        dense = keras.layers.Dense(128, activation='relu')(flatten)
        drop = keras.layers.Dropout(0.2)(dense)
        outputs = keras.layers.Dense(10, activation='softmax')(drop)
        model = keras.Model(inputs = inputs, outputs = outputs)
        return model

func_model = create_func_mondel()
func_model.summary()



# %%

# 3. Model Class Subclassing 사용
# 파이토치와 유사하게 사용

import tensorflow as tf

class SubClassModle(keras.Model):
        def __init__(self):
                super(SubClassModle, self).__init__()
                self.flatten = keras.layers.Flatten(input_shape=(28, 28))
                self.dense_1 = keras.layers.Dense(128, activation='relu')
                self.drop = keras.layers.Dropout(0.2)
                self.dense_2 = keras.layers.Dense(10, activation='softmax')

        def call(self, x, training=False):
                x = self.flatten(x)
                x = self.dense_1(x)
                x = self.drop(x)
                return self.dense_2(x)

subclass_model = SubClassModle()

inputs = tf.zeros((1,28,28))
subclass_model(inputs)
subclass_model.summary()



# %%

# 가상의 data 만들어서 예측해보기
inputs = tf.random.normal((1, 28, 28))
outputs = subclass_model(inputs)
pred = tf.argmax(outputs, -1)
print('predicted class :', pred)


# %%
