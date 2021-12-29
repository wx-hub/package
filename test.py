# -*- coding:UTF-8 -*-

import os, sys, pprint, time,h5py
import scipy.misc
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from glob import glob
from random import shuffle
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.io as io
from model import *
import scipy.io
import natsort



'''
flags = tf.app.flags
flags.DEFINE_integer("epoch", 1000, "Epoch to train ")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam ")
flags.DEFINE_integer("batch_size", 1, "The number of batch images ")
flags.DEFINE_integer("image_size", 512, "The size of image to use (will be center cropped) ")
flags.DEFINE_integer("output_size", 512, "The size of the output images to produce ")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints ")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples ")
flags.DEFINE_string("feature_dir", "features", "Directory name to save the image samples ")
flags.DEFINE_boolean("is_crop", True, "True for training, False for testing ")

FLAGS = flags.FLAGS
'''

flags = tf.app.flags
flags.DEFINE_integer("epoch", 1000, "Epoch to train ")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam ")
flags.DEFINE_integer("batch_size", 1, "The number of batch images ")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints ")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples ")

FLAGS = flags.FLAGS

def imread(path):
    f = loadmat(path)
    f = f['img']
    m = np.array(f)
    return m


def get_image(image_path):
    return imread(image_path)


def main(_):
    input_img = tf.placeholder(tf.float32, [1, 224, 224, 32, 1], name='input_img')
    g_logits = generator(input_img)
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    tl.layers.initialize_global_variables(sess)
    saver.restore(sess, './checkpoint/model.ckpt')

    imgs = []
    m = natsort.natsorted(glob('./data_test/*.mat'))#natsort元素排序

    for im in m:
        imgs.append(im)
    sample_files = np.asarray(imgs, np.chararray)
    oimg = [get_image(sample_file)for sample_file in sample_files]
    length = len(oimg)
    oimg = np.array(oimg).astype(np.float32)

    for i in range(length):
        fimg = oimg[i, :, :]
        img = fimg[np.newaxis, :, :, :, np.newaxis]
        outimg = sess.run([g_logits], feed_dict={input_img: img}) #list类型
        outimg = np.array(outimg).astype(np.float32)   
        outimg = np.squeeze(outimg)#squeeze去维度
        print(outimg.shape)
        io.savemat('./output/{}.mat'.format(i+1), {'predict': outimg})




if __name__ == '__main__':
    tf.app.run()