# -*- coding:UTF-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import inference
import os
import numpy as np



BATCH_SIZE = 64
LEARNING_RATE_BASE = 0.0001
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0005
TRAINING_STEPS = 60000
MOVING_AVERAGE_DECAY = 0.96
DROPOUT = 0.6

NUM_CHANNELS = 1
IMAGE_SIZE = 28
OUTPUT_NODE = 10

MODEL_SAVE_PATH = '/path/alex_net/model2/'
MODEL_NAME = 'model.ckpt'

Weights = {
    # 'wc1':tf.get_variable([3,3,1,64],initializer=tf.truncated_normal_initializer(stddev=0.1)),
    'wc1': tf.Variable(tf.truncated_normal([3, 3, 1, 64], stddev=0.1)),
    'wc2': tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1)),
    'wc3': tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1)),
    'wf1': tf.Variable(tf.truncated_normal([4 * 4 * 256, 1024], stddev=0.1)),
    'wf2': tf.Variable(tf.truncated_normal([1024, 1024], stddev=0.1)),
    'wo': tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
}

biases = {
    'bc1': tf.Variable(tf.zeros([64])),
    'bc2': tf.Variable(tf.zeros([128])),
    'bc3': tf.Variable(tf.zeros([256])),
    'bf1': tf.Variable(tf.zeros([1024])),
    'bf2': tf.Variable(tf.zeros([1024])),
    'fo': tf.Variable(tf.zeros([OUTPUT_NODE]))
}

def train(mnist):

    x = tf.placeholder(tf.float32,
                       [BATCH_SIZE,
                       IMAGE_SIZE,
                       IMAGE_SIZE,
                       NUM_CHANNELS],
                       name='x-input')
    y_ = tf.placeholder(tf.float32,
                        [None,OUTPUT_NODE],
                        name='y-input')


    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)

    y = inference.alex_net(X=x,Weights=Weights,biases=biases,dropout=DROPOUT)

    global_step = tf.Variable(0,trainable=False)
    #定义滑动平均
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # loss = cross_entropy_mean+tf.add_n(tf.get_collection('losses'))
    loss = cross_entropy_mean

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples/BATCH_SIZE,
        LEARNING_RATE_DECAY
    )

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    # train_step = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(loss,global_step=global_step)
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
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step = global_step)
        # xa,ya = mnist.train.images,mnist.train.labels
        # _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xa, y_: ya})
        # print('After %d training steps,loss on training set is %g' % (step, loss_value))
def main(argv = None):
    mnist = input_data.read_data_sets('/path/to/MNIST_data',one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()

