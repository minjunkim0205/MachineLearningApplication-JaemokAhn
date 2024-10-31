import tensorflow as tf

# 벡터 정의
vector01 = tf.constant([1, 2, 3])
vector02 = tf.constant([4, 5, 6])

# 내적 계산
dot_product = tf.tensordot(vector01, vector02, axes=1)

# 결과 출력
print(dot_product.numpy())