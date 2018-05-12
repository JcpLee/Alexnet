# -*- coding:UTF-8 -*-
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import inference
import train_net
import numpy as np

EVAL_INTERVAL_SECS = 60


def evaluate(mnist):
    # with tf.Graph().as_default() as g:
        # x = tf.placeholder(tf.float32, [None, mnist_inferenceLeNet.INPUT_NODE], name='x-input')
        x = tf.placeholder(tf.float32,
                           [None,
                            train_net.IMAGE_SIZE,
                            train_net.IMAGE_SIZE,
                            train_net.NUM_CHANNELS],
                           name='x-input')
        y_ = tf.placeholder(tf.float32, [None, train_net.OUTPUT_NODE], name='y-input')

        reshape_xs = np.reshape(mnist.validation.images, (mnist.validation.num_examples,
                                                          train_net.IMAGE_SIZE,
                                                          train_net.IMAGE_SIZE,
                                                          train_net.NUM_CHANNELS))

        validata_feed = {x: reshape_xs, y_: mnist.validation.labels}

        y,f = inference.alex_net(X=x,output=10,dropout=train_net.DROPOUT,regularizer=None)
        #tf.argmax()返回向量中最大值位置,tf.equal()返回两个向量对应位置比较结果 返回值为布尔类型
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        #数据类型转换
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # variable_averages = tf.train.ExponentialMovingAverage(train_net.MOVING_AVERAGE_DECAY)
        # #加载变量的滑动平均值
        # saver = tf.train.Saver(variable_averages.variables_to_restore())

        #加载保存模型的变量
        saver = tf.train.Saver()

        while True:
            with tf.Session() as sess:
                #返回模型变量取值的路径
                ckpt = tf.train.get_checkpoint_state(train_net.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    #ckpt.model_checkpoint_path返回最新的模型变量取值的路径
                    saver.restore(sess, ckpt.model_checkpoint_path)

                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    print('前层特征：')
                    print(sess.run(f,feed_dict=validata_feed))

                    print('After %s traing steps validation accuracy is %g' % (global_step, sess.run(accuracy, feed_dict=validata_feed)))
                else:
                    print('NO checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets('/path/to/MNIST_data', one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()