# -*- coding:UTF-8 -*-
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.platform import gfile
import inference
import train_net
import numpy as np


def evaluate(mnist,i):
        x = mnist.test.images[i]
        reshape_xs = np.reshape(x,(1,
                                    train_net.IMAGE_SIZE,
                                    train_net.IMAGE_SIZE,
                                    train_net.NUM_CHANNELS))

        with tf.Session() as sess:
            #返回模型变量取值的路径
            model_filename = 'model_pb/combined_model.pb'
            with gfile.FastGFile(model_filename, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                sess.graph.as_default()
                tf.import_graph_def(graph_def,name='')

            input_x = sess.graph.get_tensor_by_name('x-input:0')
            op = sess.graph.get_tensor_by_name('out:0')

            ret = sess.run(op, feed_dict={input_x:reshape_xs})
            print(ret)
            out_true = tf.argmax(mnist.test.labels[i])
            print(sess.run(out_true))


def main(argv=None):
    mnist = input_data.read_data_sets('/path/to/MNIST_data', one_hot=True)
    #输入要选择的测试的测试集下标
    evaluate(mnist,67)


if __name__ == '__main__':
    tf.app.run()