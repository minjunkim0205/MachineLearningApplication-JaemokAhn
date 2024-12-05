import tensorflow as tf

# 텐서 생성
tensor01 = tf.constant([1, 2, 3], dtype=tf.int32)
tensor02 = tf.ones([3], dtype=tf.int32)

# 덧샘 연산
result = tensor01 + tensor02

# 결과 출력
print(result, end="")