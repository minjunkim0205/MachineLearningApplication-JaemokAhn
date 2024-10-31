import tensorflow as tf

# Use | ts.cast()

# 정수형 텐서 생성
x = tf.constant([1, 2, 3, 4], dtype=tf.int32)
# 정수형 텐서를 실수형으로 변환
x_float = tf.cast(x, dtype=tf.float32)
# 부울형 텐서 생성
bool_tensor = tf.constant([True, False, True], 
dtype=tf.bool)
# 부울형 텐서를 정수형으로 
int_tensor = tf.cast(bool_tensor, dtype=tf.int32)