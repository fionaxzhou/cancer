# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2017-02-03 16:18:59
# @Last Modified by:   yancz1989
# @Last Modified time: 2017-02-22 23:34:18

import tensorflow as tf
import numpy as np
from env import *
import argparse
import os
import tensorflow.contrib.slim as slim
import sys
import time
import cv2
import json
import logging
import utils.inception_v2
from tensorflow.contrib.layers import convolution2d as conv2d
from tensorflow.contrib.layers import max_pool2d, fully_connected, avg_pool2d, dropout, flatten
from tensorflow.contrib.layers import xavier_initializer_conv2d as init_conv
from tensorflow.contrib.layers import variance_scaling_initializer as init_fc
from trunk import Trunk
from utilities import mkdir

class FPTrunk(Trunk):
  def __init__(self, data_list, x, y, bsize, sess):
    Trunk.__init__(self, data_list, x, y, bsize, sess)

  def read(self, key, idx):
    img = cv2.imread(self.data_list[key][idx][0], 0)
    # print(key, self.data_list[key][idx][0])
    if img.size != 64 * 64:
      x_ = None
    else:
      x_ = np.array(img).astype(np.float32).reshape(64, 64, 1)
    y_ = np.array(self.data_list[key][idx][1]).reshape(1)
    return x_, y_

def model(H, x, training):
  net = dropout(x, 0.5, is_training = training)
  # net = conv2d(net, 64, [3, 3], activation_fn = tf.nn.relu)
  # net = conv2d(net, 64, [3, 3], activation_fn = tf.nn.relu)
  # net = max_pool2d(net, [2, 2], padding = 'VALID')
  # net = conv2d(net, 128, [3, 3], activation_fn = tf.nn.relu)
  # net = conv2d(net, 128, [3, 3], activation_fn = tf.nn.relu)
  # net = max_pool2d(net, [2, 2], padding = 'VALID')
  # ksize = net.get_shape().as_list()
  # net = max_pool2d(net, [ksize[1], ksize[2]])
  net = fully_connected(flatten(net), 256, activation_fn = tf.nn.relu)
  net = dropout(net, 0.5, is_training = training)
  logits = fully_connected(net, 1, activation_fn = tf.nn.sigmoid)
  preds = tf.cast(tf.greater(logits, 0.5), tf.int64)
  return logits, preds

def build(H, dat, sess):
  with open(META_DIR + 'fp.json') as fpj:
    meta = json.load(fpj)
  bsize = H['batch_size']
  x = tf.placeholder(tf.float32, shape = [64, 64, 1])
  y = tf.placeholder(tf.float32, shape = [1,])
  training = tf.placeholder(tf.bool)

  fptrunk = FPTrunk(dat, x, y, bsize, sess)
  Xt, Yt = tf.train.batch(fptrunk.q['train'].dequeue(), batch_size = bsize, capacity = bsize)
  Xv, Yv = tf.train.batch(fptrunk.q['valid'].dequeue(), batch_size = bsize, capacity = bsize)

  logits, preds = model(H, Xt, training)
  loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
      logits=logits, labels=tf.cast(Yt, tf.float32)))
  varst = tf.trainable_variables()
  gstep = tf.Variable(0, trainable = False)
  opts = {
      'RMS': tf.train.RMSPropOptimizer,
      'Adam': tf.train.AdamOptimizer,
      'SGD': tf.train.GradientDescentOptimizer
    }
  opt = opts[H['opt']](learning_rate = H['lr'])
  grads_vars = opt.compute_gradients(loss, varst)
  grads = [gv[0] for gv in grads_vars]
  vars = [gv[1] for gv in grads_vars]
  capped, norm = tf.clip_by_global_norm(grads, H['norm_clip'])
  train_opt = opt.apply_gradients([(capped[i], vars[i]) for i in range(len(vars))], global_step = gstep)

  saver = tf.train.Saver(max_to_keep = None)
  return (x, y, training, Xt, Yt, Xv, Yv, logits, loss, preds, opt, varst, gstep, train_opt, saver, fptrunk)

def FPFN(Y, Y_hat):
  return (np.sum((Y != Y_hat) & (Y == 1)), np.sum((Y != Y_hat) & (Y == 0)))

def train(args):
  with open(args.hype) as f:
    H = json.load(f)
    H['subset'] = args.subset
    H['save_dir'] = FPR_DIR + 'subset' + str(H['subset'])
    mkdir(H['save_dir'])
    if args.gpu != None:
      H['gpu'] = args.gpu
  with open(META_DIR + 'fp.json') as fpj:
    meta = json.load(fpj)

  dat = {}
  dat['train'] = []
  dat['valid'] = []
  for i in range(10):
    if i == args.subset:
      dat['valid'] = meta['subset' + str(i)]
    else:
      dat['train'] += meta['subset' + str(i)]
  tf.set_random_seed(2012310818)

  os.environ['CUDA_VISIBLE_DEVICES'] = str(H['gpu'])
  gpu_options = tf.GPUOptions()
  gpu_options.allow_growth = True
  config = tf.ConfigProto(gpu_options = gpu_options)

  with tf.Session(config = config) as sess:
    (x, y, training, Xt, Yt, Xv, Yv, logits, loss, preds, opt, varst, gstep, train_opt, saver, fptrunk) = build(H, dat, sess)

    sess.run(tf.global_variables_initializer())
    fptrunk.start()
    if args.weight != None:
      logging.info('Restoring from %s...' % args.weight)
      saver.restore(sess, weight)
    bsize = fptrunk.bsize
    train_batches = fptrunk.nbatches['train']
    valid_batches = fptrunk.nbatches['valid']
    for epoch in range(H['epochs']):
      tst = time.time()
      tol_loss, tol_tfn, tol_tfp, tol_vfn, N, P, vN, vP, tol_vfp, tol_acc, tol_vacc = [0.0] * 11
      for step in range(1, train_batches):
        curX, curY = sess.run([Xt, Yt])
        _, tloss, tpreds = sess.run([train_opt, loss, preds],
            feed_dict = {Xt: curX, Yt: curY, training : True})
        fn, fp = FPFN(curY, tpreds)
        N += np.sum(curY == 0)
        P += np.sum(curY == 1)
        tol_loss += tloss
        tol_tfn += fn
        tol_tfp += fp
        tol_acc += fp + fn
        if step % 100 == 0:
          cnt = (step * bsize)
          logstr = ('Training batchs %d, avg loss %f, acc %f, FN %d/%d, FP %d/%d.' %
            (step, tol_loss / step, (cnt - tol_acc) / cnt, tol_tfn, P, tol_tfp, N))
          print(logstr)
          logging.info(logstr)

      for step in range(valid_batches):
        curX, curY = sess.run([Xv, Yv])
        curY = curY.reshape(bsize, 1)
        tpreds = sess.run(preds, feed_dict = {Xt : curX, Yt: curY, training : False})
        fn, fp = FPFN(curY, tpreds)
        vN += np.sum(curY == 0)
        vP += np.sum(curY == 1)
        tol_vfn += fn
        tol_vfp += fp
        tol_vacc += fn + fp

      t = time.time() - tst
      logstr = ('epoch %d, time elapse %f, training loss %f,' +
        ' valid avg FN %f, FP %f, acc %f.') % (epoch + 1, t,
        tol_loss / train_batches, float(tol_vfn) / vP, float(tol_vfp) / vN, tol_vacc / valid_batches)

      print(logstr)
      logging.info(logstr)
      saver.save(sess, H['save_dir'] + '/save.ckpt', global_step = gstep)

    logging.info('training finished, try ending...')
    fptrunk.stop()
    logging.info('ended...')
    sess.close()

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--hype', default = None, type = str)
  parser.add_argument('--weight', default = None, type = str)
  parser.add_argument('--subset', default = None, type = int)
  parser.add_argument('--train', default = 1, type = int)
  parser.add_argument('--gpu', default = None, type = int)
  args = parser.parse_args()
  logging.basicConfig(filename = 'fp_detect_' + str(args.subset) + '.log',
        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)
  if args.hype == None or args.subset == None:
    raise 'Error: Config file and subset need to be specified.'
  if args.train == 1:
    train(args)

if __name__ == '__main__':
  main()