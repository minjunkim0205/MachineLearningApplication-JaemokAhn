import tensorflow as tf
import numpy as np

# Use | tf.convert_to_tensor()

m1=[[1.0, 2.0], [3.0, 4.0]]
m2=np.array([[1.0, 2.0],
             [3.0, 4.0]], dtype=np.float32)
t1=tf.convert_to_tensor(m1, dtype=tf.float32)
t2=tf.convert_to_tensor(m2, dtype=tf.float32)
print(t1)
print(t2)
