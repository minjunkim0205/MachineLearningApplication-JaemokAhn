import tensorflow as tf
import numpy as np

# 1차원 텐서 생성
tensor_1d = tf.constant([1, 2, 3, 4, 5, 6], dtype=tf.int32)
# 1차원 텐서를 2x3의 2차원 텐서로 변환
tensor_2d = tf.reshape(tensor_1d, shape=(2, 3))
print("[ tensor_2d ]", end="\n")
print(tensor_2d, end="\n\n\n")


# 2차원 텐서 생성
tensor_2d = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int32)
# 2차원 텐서를 3차원 텐서로 변환
tensor_3d = tf.reshape(tensor_2d, shape=(2, 1, 3))#((3, 1, 3) >> 1행 3열이 2개)
print("[ tensor_3d ]", end="\n")
print(tensor_3d, end="\n\n\n")


# 예시로, 8개의 샘플이 있고 각 샘플은 3개의 피처를 가짐
tensor_batch = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], 
                            [13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]], dtype=tf.int32)
# 배치 크기를 4로 변경 (2x4 배치로 변환)
reshaped_batch = tf.reshape(tensor_batch, shape=(4, 2, 3))
print("[ reshaped_batch ]", end="\n")
print(reshaped_batch, end="\n\n\n")