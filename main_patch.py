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
import pandas as pd
import matplotlib.pyplot as plt

flags = tf.app.flags
flags.DEFINE_integer("epoch",1000, "Epoch to train ")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam ")
flags.DEFINE_integer("batch_size", 10, "The number of batch images ")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints ")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples ")
flags.DEFINE_integer("patch_size1", 96, "patch size to image ")
flags.DEFINE_integer("patch_size2", 32, "patch size to image ")

FLAGS = flags.FLAGS


def random_crop(num, input_1, input_2, size1,size2):
    k = 0
    inp_1 = []
    inp_2 = []

    while(k < num):
        assert input_1.shape[0] > size1 and input_1.shape[0] > size2
        output_top = random.randint(0, input_1.shape[0] - size1)
        output_bottom = output_top + size1
        output_left = random.randint(0, input_1.shape[1] - size1)
        output_right = output_left + size1
        output_up = random.randint(0, input_1.shape[2] - size2)
        output_down = output_up + size2
        input1 = input_1[output_top:output_bottom, output_left:output_right, output_up:output_down]
        input2 = input_2[output_top:output_bottom, output_left:output_right, output_up:output_down]

        judge_zero = np.all(input2 == 0)
        if judge_zero:
            continue
        else:
            k = k + 1
            inp_1.append(input1)
            inp_2.append(input2)
        patches_img = np.array(inp_1)
        patches_mask=np.array(inp_2)

    return patches_img, patches_mask


def loadimg(path):
    return scipy.io.loadmat(path)['img']


def loadlabel(path):
    return scipy.io.loadmat(path)['label']


def main(_):
    # ctrain=0
    tl.files.exists_or_mkdir(FLAGS.checkpoint_dir)
    ## 占位符声明 tf.placeholder(dtype, shape=None, name=None)用于传入外部数据
    input_img = tf.placeholder(tf.float32, [FLAGS.batch_size, 96, 96, 32, 1], name='input_img')
    label_img = tf.placeholder(tf.float32, [FLAGS.batch_size, 96, 96, 32, 1], name='label_img')
    g_logits = generator(input_img)
    g_loss = 1 - tl.cost.dice_coe(g_logits, label_img)
    ## adam优化器，beta为0.5
    g_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1).minimize(g_loss)
    # g_optim = tf.train.GradientDescentOptimizer(0.01).minimize(g_loss)
    # ## 开启会话
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    tl.layers.initialize_global_variables(sess)
    ## 数据集读取
    imgs = []
    for im in glob('./data_train/*.mat'):
        imgs.append(im)
    origin_files = np.asarray(imgs, np.chararray)
    iter_counter = 0
    lossG = []
    for epoch in range(FLAGS.epoch):
        ## shuffle data打乱数据
        shuffle(origin_files)
        ## load image data
        batch_idxs = len(origin_files)
        # batch_idxs = len(origin_files) // FLAGS.batch_size
        for idx in range(0, batch_idxs):
            start_time = time.time()
            batch_files = origin_files[idx :(idx + 1) ]
            # batch_files = origin_files[idx * FLAGS.batch_size:(idx + 1) * FLAGS.batch_size]
            ## get real images
            batch = [loadimg(batch_file) for batch_file in batch_files]
            batch2 = [loadlabel(batch_file) for batch_file in batch_files]
            batch_a = np.array(batch).astype(np.float32)
            batch_c = np.array(batch2).astype(np.float32)
            batcha=np.squeeze(batch_a)
            batchc=np.squeeze(batch_c)
            # batch_b=(batcha-np.mean(batcha))/np.std(batcha)
            patch_imgs,patch_masks=random_crop(FLAGS.batch_size,batcha,batchc,FLAGS.patch_size1,FLAGS.patch_size2)
            patch_imgs = patch_imgs[:, :, :, :, np.newaxis]
            patch_masks=patch_masks[:, :, :, :, np.newaxis]
            # a = patch_imgs[0, :, :, :, 0]
            # plt.figure()
            # for k in range(32):
            #     a1 = a[ :, :, k]
            #     plt.subplot(4, 8, k + 1)
            #     plt.imshow(a1, cmap='gray')
            # plt.show()
            ## train model
            t_img, errG, _ = sess.run([g_logits, g_loss, g_optim], feed_dict={input_img: patch_imgs, label_img: patch_masks})
            print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, gloss: %.5f" \
                  % (epoch, FLAGS.epoch, idx, batch_idxs, time.time() - start_time, errG))
            lossG.append(errG)
            scipy.io.savemat('./results/total_loss.mat', {'lossG': lossG})

    ## valid and save

        saver.save(sess, './checkpoint/model.ckpt')
        MODEL_DIR="loss"
        loss = {'lossG': lossG}
        loss_dir = os.path.join(MODEL_DIR, "loss.csv")  # 连接两个或更多的路径名组件:loss\loss.csv
        pd.DataFrame(loss).to_csv(loss_dir, index=False)



if __name__ == '__main__':
    tf.app.run()
