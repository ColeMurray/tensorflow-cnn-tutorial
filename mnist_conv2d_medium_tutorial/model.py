import tensorflow as tf


class Model(object):
    def __init__(self, batch_size=128, learning_rate=1e-4, num_labels=10):
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._num_labels = num_labels

    def inference(self, images):
        pass

    def train(self, loss, global_step):
        pass

    def loss(self, logits, labels):
        pass

    def accuracy(self, logits, y):
        with tf.variable_scope('accuracy') as scope:
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1)), dtype=tf.float32),
                                      name=scope.name)
            tf.summary.scalar('accuracy', accuracy)
        return accuracy

    def _create_conv2d(self, x, W):
        return tf.nn.conv2d(input=x,
                            filter=W,
                            strides=[1, 1, 1, 1],
                            padding='SAME')

    def _create_max_pool_2x2(self, input):
        return tf.nn.max_pool(value=input,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

    def _create_weights(self, shape):
        return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1, dtype=tf.float32))

    def _create_bias(self, shape):
        return tf.Variable(tf.constant(1., shape=shape, dtype=tf.float32))

    def _activation_summary(self, x):
        tensor_name = x.op.name
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
