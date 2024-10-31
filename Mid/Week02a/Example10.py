import tensorflow as tf

nspikes = tf.Variable([False] * 8, name='spikes')
nspikes.numpy()

new_checkpoint = tf.train.Checkpoint(spikes=nspikes)
new_checkpoint.restore("./Mid/Week02a/Example09Checkpoint/spikes.ckpt-1")

result = nspikes.numpy()
print(result)