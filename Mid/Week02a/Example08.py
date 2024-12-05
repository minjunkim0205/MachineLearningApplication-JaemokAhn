import tensorflow as tf

# 데이터 스파이크 5 이상을 검출하는 코드(cur 데이터가 prev 데이터 기준으로)

raw_data = [1.0, 2.0, 8.0, -1.0, 0, 5.5, 6.0, 13.0]
spikes = tf.Variable([False] * len(raw_data), name="spikes")
spikes.numpy()

for i in range(1, len(raw_data)):
    if raw_data[i] - raw_data[i-1] > 5:
        spikes_val = spikes.numpy()
        spikes_val[i] = True
        spikes.assign(spikes_val)

print(spikes.numpy(), end="\n")