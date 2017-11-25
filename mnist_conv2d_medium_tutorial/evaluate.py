import tensorflow as tf

from mnist_conv2d_medium_tutorial import mnist
from mnist_conv2d_medium_tutorial.model import Model

FLAGS = tf.app.flags.FLAGS


def evaluate():
    with tf.Graph().as_default():
        images, labels = mnist.load_test_data(FLAGS.test_data)
        model = Model()

        logits = model.inference(images, keep_prob=1.0)
        accuracy = model.accuracy(logits, labels)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver.restore(sess, FLAGS.checkpoint_file_path)

            total_accuracy = sess.run([accuracy])
            print('Test accuracy: {}'.format(total_accuracy))


def main(argv=None):
    evaluate()


if __name__ == '__main__':
    tf.app.flags.DEFINE_string('checkpoint_file_path', 'checkpoints/model.ckpt-10000-10000', 'path to checkpoint file')
    tf.app.flags.DEFINE_string('test_data', 'data/mnist_test.csv', 'path to test data')

    tf.app.run()
