# -*- coding:UTF-8 -*-
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import inference
import train_net
import numpy as np


def evaluate(mnist,i):
        x = tf.placeholder(tf.float32,
                           [1,
                            train_net.IMAGE_SIZE,
                            train_net.IMAGE_SIZE,
                            train_net.NUM_CHANNELS],
                           name='x-input')

        y,f = inference.alex_net(X=x,output=10,dropout=train_net.DROPOUT,regularizer=None)

        result = tf.argmax(y,1)

        variable_averages = tf.train.ExponentialMovingAverage(train_net.MOVING_AVERAGE_DECAY)
        #加载变量的滑动平均值
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        #加载保存模型的变量
        # saver = tf.train.Saver()

        with tf.Session() as sess:
                #返回模型变量取值的路径
            ckpt = tf.train.get_checkpoint_state(train_net.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                    #ckpt.model_checkpoint_path返回最新的模型变量取值的路径
                saver.restore(sess, ckpt.model_checkpoint_path)

                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                reshape_xs = np.reshape(mnist.test.images[i], (1,
                                                                   train_net.IMAGE_SIZE,
                                                                   train_net.IMAGE_SIZE,
                                                                   train_net.NUM_CHANNELS))

                test_feed = {x: reshape_xs}

                result = sess.run(result, feed_dict=test_feed)

                print('After %s traing steps test result is %g' % (global_step, result))
                print('truly result is %s'% (sess.run(tf.argmax(mnist.test.labels[i]))))
            else:
                print('NO checkpoint file found')
                return


def main(argv=None):
    mnist = input_data.read_data_sets('/path/to/MNIST_data', one_hot=True)
    #输入要选择的测试的测试集下标
    evaluate(mnist,49)


if __name__ == '__main__':
    tf.app.run()