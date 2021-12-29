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
flags.DEFINE_integer("epoch", 800, "Epoch to train ")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam ")
flags.DEFINE_integer("batch_size", 1, "The number of batch images ")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints ")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples ")
flags.DEFINE_integer("patch_size1",96,"Patch to images")
flags.DEFINE_integer("patch_size2",32,"Patch to images")
flags.DEFINE_integer("stride_size1",48,"Stride to images")
flags.DEFINE_integer("stride_size2",16,"Stride to images")
FLAGS = flags.FLAGS

def pred_to_img(pred):
    pred_imgs=np.empty((pred.shape[0], pred.shape[1], pred.shape[2]))
    # for pix in range (pred.shape[0]):
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            for k in range(pred.shape[2]):
                if pred[ i, j, k] >= 0.5:
                    pred_imgs[ i, j, k] = 1
                else:
                    pred_imgs[ i, j, k] = 0
                        # pred_imgs=np.reshape(pred_imgs,(pred.shape[0],size,size,size,1))

# def pred_to_imgs(pred, size=FLAGS.patch):
#     pred_images = np.empty((pred.shape[0], size, size, size, 2))
#     for pix in range(pred.shape[0]):
#         for i in range(pred.shape[1]):
#            for j in range(pred.shape[2]):
#                 for k in range(pred.shape[3]):
#                     if pred[pix, i, j, k, 0] >= 0.5:
#                         pred_images[pix, i, j, k, 0] = 1
#                         if pred[pix, i, j, k, 1] >= 0.5:
#                             pred_images[pix, i, j, k, 1] = 1
#                         else:
#                             pred_images[pix, i, j, k, 1] = 0
#                     else:
#                         pred_images[pix, i, j, k, 0] = 0
#
#     pred_images = np.reshape(pred_images, (pred_images.shape[0], size, size, size, 2))
#     return pred_images
    return pred_imgs



def recompone_overlap(preds, img_h, img_w, img_d, stride1, stride2):
    patch_h = preds.shape[1]
    patch_w = preds.shape[2]
    patch_d= preds.shape[3]
    N_patches_h = (img_h-patch_h)//stride1+1
    N_patches_w = (img_w-patch_w)//stride1+1
    N_patches_d = (img_d-patch_d)//stride2+1
    N_patches_img = N_patches_h * N_patches_w*N_patches_d
    N_full_imgs = preds.shape[0]//N_patches_img
    full_prob = np.zeros((N_full_imgs,img_h,img_w,img_d))  #itialize to zero mega array with sum of Probabilities
    full_sum = np.zeros((N_full_imgs,img_h,img_w,img_d))

    k = 0 #iterator over all the patches
    for i in range(N_full_imgs):
        for h in range((img_h - patch_h) // stride1 + 1):
            for w in range((img_w - patch_w) // stride1 + 1):
                for d in range((img_d-patch_d)//stride2+1):
                    full_prob[i,h*stride1:(h*stride1)+patch_h,w*stride1:(w*stride1)+patch_w,d*stride2:(d*stride2)+patch_d]+=preds[k]
                    full_sum[i,h*stride1:(h*stride1)+patch_h,w*stride1:(w*stride1)+patch_w,d*stride2:(d*stride2)+patch_d]+=1
                    k+=1

    # a = full_prob[0]
    # b = np.squeeze(a)
    # plt.figure()
    # for k in range(32):
    #     a1 = b[:, :, k]
    #     plt.subplot(4, 8, k + 1)
    #     plt.imshow(a1, cmap='gray')
    # plt.show()
    full_prob1 = np.squeeze(full_prob)
    full_sum1=np.squeeze(full_sum)
    final_avg = full_prob1/full_sum1
    return final_avg

# def border_overlap(img, size1, stride1):
#     height = img.shape[0]
#     width = img.shape[1]
#     leftover_h = (height-size1)%stride1
#     leftover_w = (width - size1) % stride1
#     if (leftover_h != 0):
#         img1=np.zeros((height+(stride1-leftover_h), width, img.shape[2]))
#         img1[0:height,0:width,0:img.shape[2]]=img
#         img=img1
#     if (leftover_w != 0):
#         img1 = np.zeros((height, width+(stride1-leftover_w), img.shape[2]))
#         img1[0:img.shape[0], 0:width, 0:img.shape[2]] = img
#         img=img1
#     return img

def  generate_patches_imgs(x, size1 , size2, stride1, stride2):
    height = x.shape[0]
    width = x.shape[1]
    depth = x.shape[2]
    patches = []
    for i in range((height - size1) //stride1 + 1):
        for j in range((width-size1) //stride1+1):
            for k in range((depth-size2)//stride2+1):
                patch = x[i*stride1:(i*stride1)+size1, j*stride1:(j*stride1)+size1, k*stride2:(k*stride2)+size2]
                patches.append(patch)
    patches = np.array(patches)
    return patches

def imread(path):
    f = loadmat(path)
    f = f['img']
    m = np.array(f)
    return m


def get_image(image_path):
    return imread(image_path)


def main(_):
    input_img = tf.placeholder(tf.float32, [None, 96, 96, 32, 1], name='input_img')
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
        # fimg=border_overlap(fimg, FLAGS.patch_size1, FLAGS.stride_size1)
        fimg1=generate_patches_imgs(fimg, FLAGS.patch_size1, FLAGS.patch_size2, FLAGS.stride_size1, FLAGS.stride_size2)
        img = fimg1[:, :, :, :, np.newaxis]
        outimg = sess.run([g_logits], feed_dict={input_img: img}) #list类型
        outimg = np.array(outimg).astype(np.float32)
        outimg = np.squeeze(outimg)#squeeze去维度
        pred_imgs = recompone_overlap(outimg, fimg.shape[0], fimg.shape[1], fimg.shape[2], FLAGS.stride_size1,  FLAGS.stride_size2)
        pred_imgs1 = pred_to_img(pred_imgs)
        # pred_imgs = np.squeeze(pred_imgs1)
        print(pred_imgs1.shape)
        io.savemat('./output/{}.mat'.format(i+1), {'predict': pred_imgs1})


if __name__ == '__main__':
    tf.app.run()