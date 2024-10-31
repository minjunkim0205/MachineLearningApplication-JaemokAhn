import tensorflow as tf

x = tf.constant([[1, 2]])
y = tf.constant([[1, 2]])

negMatrix = tf.negative(x)
print(negMatrix)
negMatrix.numpy()
z=tf.add(x, y)
tf.subtract(x, y)
tf.multiply(x, y)
tf.pow(x, y)
tf.exp(1.1) #float지정
tf.sqrt(0.3) #float지정
tf.divide(x, y)
tf.truediv(x, y)
tf.math.floordiv(x, y)
tf.math.mod(x, y)
print(z.numpy(), end="\n")

tensor1 = tf.constant([1, 2, 3],dtype=tf.int32) 
tensor2 = tf.ones([3],dtype=tf.int32)

result = tensor1 + tensor2
print(result, end="\n")