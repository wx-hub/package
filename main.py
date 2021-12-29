# -*- coding:UTF-8 -*-

import os, sys, pprint, time
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from glob import glob
from random import shuffle
import random
import scipy.misc
from model import *
import scipy.io


flags = tf.app.flags
flags.DEFINE_integer("epoch", 1000, "Epoch to train ")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam ")
flags.DEFINE_integer("batch_size", 1, "The number of batch images ")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints ")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples ")

FLAGS = flags.FLAGS


def loadimg(path):
    return scipy.io.loadmat(path)['img']

def loadlabel(path):
    return scipy.io.loadmat(path)['label']


def main(_):
    certain=0
    tl.files.exists_or_mkdir(FLAGS.checkpoint_dir)
    ## 占位符声明 tf.placeholder(dtype, shape=None, name=None)用于传入外部数据
    input_img = tf.placeholder(tf.float32, [FLAGS.batch_size, 224, 224, 32, 1], name='input_img')
    label_img = tf.placeholder(tf.float32, [FLAGS.batch_size, 224, 224, 32, 2], name='label_img')
    g_logits = generator(input_img)
    g_loss = 1 - tl.cost.dice_coe(g_logits, label_img)
    ## adam优化器，beta为0.5
    g_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1).minimize(g_loss)
    #g_optim = tf.train.GradientDescentOptimizer(0.01).minimize(g_loss) 
    # ## 开启会话
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    sess = tf.InteractiveSession(config=config)
    tl.layers.initialize_global_variables(sess)
    ## 数据集读取
    imgs = []
    for im in glob('./data_train/*.mat'):
        imgs.append(im)
    origin_files=np.asarray(imgs,np.chararray)


    lossG=[]
    for epoch in range(FLAGS.epoch):
        ## shuffle data打乱数据
        shuffle(origin_files)

        ## load image data
        batch_idxs = len(origin_files) // FLAGS.batch_size
        for idx in range(0, batch_idxs):
            start_time = time.time()
            batch_files = origin_files[idx*FLAGS.batch_size:(idx+1)*FLAGS.batch_size]

            ## get real images
            batch = [loadimg(batch_file) for batch_file in batch_files ]
            batch2 = [loadlabel(batch_file) for batch_file in batch_files]
            batch_a = np.array(batch).astype(np.float32)
            batch_c = np.array(batch2).astype(np.float32)
            batch_a=batch_a[:,:,:,:,np.newaxis]
            ## train model

            t_img, errG, _ = sess.run([g_logits, g_loss, g_optim], feed_dict={input_img: batch_a, label_img: batch_c})
            # sess.run(fetch,feed_dict):fetch:让fetch节点动起来，要fetch节点的输出，可以是list或tensor；feed_dict：替换原图中的某个tensor的值或设置graph的输入值
            # feed_dict的作用是给使用placeholder创建出来的tensor赋值
            print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, gloss: %.5f"\
                    % (epoch,FLAGS.epoch, idx, batch_idxs, time.time() - start_time, errG))
            lossG.append(errG)

            scipy.io.savemat('./results/total_loss.mat', {'lossG': lossG})
## valid and save

    saver.save(sess, './checkpoint/model.ckpt')


if __name__ == '__main__':
    tf.app.run()
    
    
   
