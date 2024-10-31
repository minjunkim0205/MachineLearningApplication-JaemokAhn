import tensorflow as tf

# 2차원 텐서 생성 예제
x = tf.constant([[1,2,3,4],
                 [5,6,7,8],
                 [9,10,11,12]], shape=(3,4), dtype=tf.int32)    

print(x, end="\n")

# 상수 예제
constant_a = tf.constant(2)
constant_b = tf.constant(3)
c = constant_a + constant_b
print(c, end="\n")

# 변수 예제
variable_a = tf.Variable(initial_value=2)
variable_b = tf.Variable(initial_value=3)
c = variable_a + variable_b
print(c, end="\n")