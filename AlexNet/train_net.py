# -*- coding:UTF-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import graph_util
import tensorflow as tf
import inference
import os
import numpy as np



BATCH_SIZE = 64
LEARNING_RATE_BASE = 0.0001
LEARNING_RATE_DECAY = 0.96
REGULARAZTION_RATE = 0.0005
TRAINING_STEPS = 60000
MOVING_AVERAGE_DECAY = 0.99
DROPOUT = 0.8

NUM_CHANNELS = 1
IMAGE_SIZE = 28
OUTPUT_NODE = 10

MODEL_SAVE_PATH = 'model/'
MODEL_NAME = 'model.ckpt'


def train(mnist):

    #定义预输入
    x = tf.placeholder(tf.float32,
                       [None,
                       IMAGE_SIZE,
                       IMAGE_SIZE,
                       NUM_CHANNELS],
                       name='x-input')
    y_ = tf.placeholder(tf.float32,
                        [None,OUTPUT_NODE],
                        name='y-input')

    #定义正则化
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)

    #调用神经网计算输出结果
    y,_ = inference.alex_net(X=x,output=OUTPUT_NODE,dropout=DROPOUT,regularizer=None)
    result = tf.argmax(y,1,name='out')

    #定义统计训练轮数的全局变量
    global_step = tf.Variable(0,trainable=False)
    #定义滑动平均
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # loss = cross_entropy_mean+tf.add_n(tf.get_collection('losses'))
    loss = cross_entropy_mean
    #定义学习率变化
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples/BATCH_SIZE,
        LEARNING_RATE_DECAY
    )

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    # train_step = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(loss,global_step=global_step)
    #同时更新滑动平均和网络参数
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            reshape_xs = np.reshape(xs,(BATCH_SIZE,
                                        IMAGE_SIZE,
                                        IMAGE_SIZE,
                                        NUM_CHANNELS))
            _,loss_value,step = sess.run([train_op,loss,global_step],feed_dict={x:reshape_xs,y_:ys})

            if i%100 == 0:
                print('After %d training steps,loss on training batch is %g'%(step,loss_value))
                #保存checkpoint文件
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step = global_step)
                #保存pb文件
                output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['out'])
                with tf.gfile.GFile('model_pb/combined_model.pb', 'wb') as f:
                    f.write(output_graph_def.SerializeToString())

def main(argv = None):
    mnist = input_data.read_data_sets('/path/to/MNIST_data',one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()

