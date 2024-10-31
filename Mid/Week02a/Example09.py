import tensorflow as tf
import os

raw_data = [1.0, 2.0, 8.0, -1.0, 0, 5.5, 6.0, 13.0]
spikes = tf.Variable([False] * len(raw_data), name="spikes")
spikes.numpy()

for i in range(1, len(raw_data)):
    if raw_data[i] - raw_data[i-1] > 5:
        spikes_val = spikes.numpy()
        spikes_val[i] = True
        spikes.assign(spikes_val)

print(spikes.numpy(), end="\n")

directory = "./Mid/Week02a/Example09Checkpoint"
if not os.path.exists(directory):
    os.makedirs(directory)
checkpoint = tf.train.Checkpoint(spikes=spikes)
save_path = checkpoint.save("./Mid/Week02a/Example09Checkpoint/spikes.ckpt")