# -*- coding:UTF-8 -*-
import tensorflow as tf
import numpy as np

#定义卷积操作函数
def conv2d(name,x,weights,bias):
    con = tf.nn.conv2d(x,weights,strides=[1,1,1,1],padding='SAME')
    rel = tf.nn.relu(tf.nn.bias_add(con,bias),name=name)
    return rel
#定义池化操作函数
def max_pool(name,x,k):
    return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME',name=name)
#定义归一化操作函数
def norm(name,x,lsize=4):
    return tf.nn.lrn(x,lsize,bias=1.0,alpha=0.001/9,beta=0.75,name=name)


#定义Alex网络
def alex_net(X,output,dropout,regularizer):
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
        'fo': tf.Variable(tf.zeros([output]))
    }

    #构造第一个卷积层
    conv1 = conv2d('conv1',X,Weights['wc1'],biases['bc1'])
    pool1 = max_pool('pool1',conv1,k=2)
    norm1 = norm('norm1',pool1,lsize=4)
    # if regularizer!=None:
    #     tf.add_to_collection('losses', regularizer(Weights['wc1']))
    drop1 = tf.nn.dropout(norm1,dropout)

    #构造第二个卷积层
    conv2 = conv2d('conv2',drop1,Weights['wc2'],biases['bc2'])
    pool2 = max_pool('pool2',conv2,k=2)
    norm2 = norm('norm2',pool2,lsize=4)
    # if regularizer!=None:
    #     tf.add_to_collection('losses', regularizer(Weights['wc2']))
    drop2 = tf.nn.dropout(norm2,dropout)

    #够造第三个卷积层
    conv3 = conv2d('conv3',drop2,Weights['wc3'],biases['bc3'])
    pool3 = max_pool('pool3',conv3,k=2)
    norm3 = norm('norm',pool3,lsize=4)
    # if regularizer!=None:
    #     tf.add_to_collection('losses', regularizer(Weights['wc3']))
    drop3 = tf.nn.dropout(norm3,dropout)

    reshaped = tf.reshape(drop3,[-1,Weights['wf1'].get_shape().as_list()[0]])

    #构造第一个全连接层
    fc1 = tf.nn.relu(tf.matmul(reshaped,Weights['wf1'])+biases['bf1'],name='fc1')
    # if regularizer!=None:
    #     tf.add_to_collection('losses', regularizer(Weights['wf1']))
    #构造第二个全连接层
    fc2 = tf.nn.relu(tf.matmul(fc1,Weights['wf2'])+biases['bf2'],name='fc2')
    # if regularizer!=None:
    #     tf.add_to_collection('losses', regularizer(Weights['wf2']))
    #构造输出层
    result = tf.matmul(fc2,Weights['wo']+biases['fo'])
    # if regularizer!=None:
    #     tf.add_to_collection('losses', regularizer(Weights['wo']))

    return result,fc2

