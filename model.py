# -*- coding:UTF-8 -*-
import tensorflow as tf
import numpy as np

#batch_normalization 批归一化，减少梯度消失
def batchnorm(inputs,is_train=False,act=None,name='batchnorm'):
    if act==None: 
        return tf.layers.batch_normalization(inputs, training=is_train)
    else:
        return act(tf.layers.batch_normalization(inputs, training=is_train))
def concat(inputs,concat_dim,name):
    try: # TF1.0
        outputs = tf.concat(inputs, concat_dim, name=name)
    except: # TF0.12
        outputs = tf.concat(concat_dim, inputs, name=name)
    return outputs
def conv3d(inputs,kernel_num=32,kernel_size=[3,3,3],is_train=False,name='conv3d'):
    with tf.variable_scope(name):
        net = tf.layers.conv3d(inputs, kernel_num, kernel_size, padding='same', name='conv')
        net = batchnorm(net, act=tf.nn.relu, is_train=is_train, name='bn')
    return net


     #is_train当它为True，代表是训练过程，这时会不断更新样本集的均值与方差。当测试时，要设置成False，这样就会使用训练样本集的均值和方差。
def generator(x, is_train=True, reuse=False):
    with tf.variable_scope("u_net", reuse=reuse):
        conv1_1 = conv3d(x, 24, is_train=is_train, name='conv1_1')
        conv1_2 = conv3d(conv1_1, 24, is_train=is_train, name='conv1_2')
        pool1 = tf.layers.max_pooling3d(conv1_2, pool_size=2,strides=2, name='pool1')
       # concatenated
        conv2_1 = conv3d(pool1, 48, is_train=is_train, name='conv2_1')
        conv2_2 = conv3d(conv2_1, 48, is_train=is_train, name='conv2_2')
        pool2 = tf.layers.max_pooling3d(conv2_2, pool_size=2,strides=2, name='pool2')
        conv3_1 = conv3d(pool2, 96, is_train=is_train, name='conv3_1')
        conv3_2 = conv3d(conv3_1, 96, is_train=is_train, name='conv3_2')
        pool3 = tf.layers.max_pooling3d(conv3_2, pool_size=2,strides=2, name='pool3')
        conv4_1 = conv3d(pool3, 192, is_train=is_train, name='conv4_1')
        conv4_2 = conv3d(conv4_1, 192, is_train=is_train, name='conv4_2')
        pool4 = tf.layers.max_pooling3d(conv4_2, pool_size=2,strides=2, name='pool4')
        conv5_1 = conv3d(pool4, 384, is_train=is_train, name='conv5_1')
        conv5_2 = conv3d(conv5_1, 384, is_train=is_train, name='conv5_2')


        up4 = tf.layers.conv3d_transpose(conv5_2,384,(2,2,2),(2,2,2),'same',name='deconv4')
        up4 = concat([up4, conv4_2] , 4, name='concat4')
        conv4_1 = conv3d(up4, 192, is_train=is_train, name='uconv4_1')
        conv4_2 = conv3d(conv4_1, 192, is_train=is_train, name='uconv4_2')  
        up3 = tf.layers.conv3d_transpose(conv4_2,192,(2,2,2),(2,2,2),'same',name='deconv3')
        up3 = concat([up3, conv3_2] , 4, name='concat3')
        conv3_1 = conv3d(up3, 96, is_train=is_train, name='uconv3_1')
        conv3_2 = conv3d(conv3_1, 96, is_train=is_train, name='uconv3_2')  
        up2 = tf.layers.conv3d_transpose(conv3_2,96,(2,2,2),(2,2,2),'same',name='deconv2')
        up2 = concat([up2, conv2_2] , 4, name='concat2')
        conv2_1 = conv3d(up2, 48, is_train=is_train, name='uconv2_1')
        conv2_2 = conv3d(conv2_1, 48, is_train=is_train, name='uconv2_2')  
        up1 = tf.layers.conv3d_transpose(conv2_2,48,(2,2,2),(2,2,2),'same',name='deconv1')
        up1 = concat([up1, conv1_2] , 4, name='concat1')
        conv1_1 = conv3d(up1, 24, is_train=is_train, name='uconv1_1')
        conv1_2 = conv3d(conv1_1, 24, is_train=is_train, name='uconv1_2')  

        out = tf.layers.conv3d(conv1_2,2,[1,1,1],padding='same',activation=tf.nn.sigmoid,name='out')
  
        
    return  out

