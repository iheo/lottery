import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt


def get_sparse_vec(dim, loc):
    x = np.zeros(dim, dtype=np.int32)
    x[loc] = 1
    return x


def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

data = []
with open('winnums-text.txt', 'r') as f:
    for i, line in enumerate(f):
        if i > 0:
            one_line = line.split()
            nums= np.array(one_line[1:6], dtype=np.int32)
            nums.sort()
            data.append(nums-1)
data = np.array(data, dtype=np.int32)
test_data = data[:4]    # most recent draws
data = data[4:]

print('size of data:%d', len(data))

# data pair (input, label) creation
x_train = []
y_train = []
dim = 69

# Inter-relations
for nums in data:
    for num in nums:
        tmp_x = get_sparse_vec(dim, nums)
        tmp_x[num] = 0

        tmp_y = get_sparse_vec(dim, num)

        x_train.append(tmp_x)
        y_train.append(tmp_y)

# sequential one by one
for i in xrange(5, len(data)):
    for k in xrange(5):
        tmp_x = get_sparse_vec(dim, data[i-5:i-1, k])
        tmp_y = get_sparse_vec(dim, data[i, k])

        x_train.append(tmp_x)
        y_train.append(tmp_y)


# # Sequential relation
# for i in xrange(1, len(data)):
#     tmp_x = get_sparse_vec(dim, data[i-1])
#     tmp_y = get_sparse_vec(dim, data[i])
#
#     x_train.append(tmp_x)
#     y_train.append(tmp_y)

x_train = np.array(x_train)
y_train = np.array(y_train)


print('Train data size: %d' % len(x_train))


# make a model
layer_sizes = [dim, 20, dim]
activations = [tf.identity, tf.nn.relu, tf.identity]

# model creation
x_input = tf.placeholder(tf.float32, shape=[None, dim])
y_label = tf.placeholder(tf.float32, shape=[None, dim])

num_layers = len(layer_sizes)
outputs = x_input
for k in xrange(1, num_layers):
    dim_in = layer_sizes[k-1] if k > 0 else dim
    dim_out = layer_sizes[k] if k < num_layers-1 else dim
    W = tf.get_variable('W%d' % k, shape=[dim_in, dim_out],
                         initializer=tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32))
    b = tf.get_variable('b%d' % k, shape=[dim_out],
                         initializer=tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32))
    outputs = tf.matmul(outputs, W) + b
    outputs = activations[k-1](outputs)

out_layer = outputs
y_predict = tf.nn.log_softmax(out_layer, name='y_predict')

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=out_layer), name='loss')
optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.01, name='optimizer').minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

num_trainables = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
print 'number of trainable weights and biases %d' % num_trainables
print 'number of data per coef : %.3f' % (1.0*len(x_train)/num_trainables)

num_epoch = 10
batch_size = 2
for epoch in xrange(num_epoch):
    # shuffle data
    sidx = np.array(range(len(x_train)))
    np.random.shuffle(sidx)

    x_train = x_train[sidx]
    y_train = y_train[sidx]

    lss = 0
    for i in range(0, len(x_train), batch_size):
        batch_x = x_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]

        _, c = sess.run([optimizer, loss], feed_dict={x_input:batch_x, y_label:batch_y})
        lss += c
    print 'epoch %d, loss : %.3f' % (epoch, c)


# Prediction
yy = np.zeros(dim)
for i in xrange(5):
    tmp = test_data[:4, i]
    x_in = get_sparse_vec(dim, tmp)
    y_pred = sess.run(y_predict, feed_dict={x_input:[x_in]})
    yy += y_pred[0]

num1 = largest_indices(yy, 5)

for y in num1:
    x_in = get_sparse_vec(dim, num1)
    x_in[y] = 0
    pred = sess.run(y_predict, feed_dict={x_input:[x_in]})
    yy += pred[0]

num2 = largest_indices(yy, 5)

print 'Most probable numbers: %s' % num1
