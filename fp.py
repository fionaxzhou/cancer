# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2017-02-03 16:18:59
# @Last Modified by:   yancz1989
# @Last Modified time: 2017-02-17 08:47:28

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

class FPTrunk(Trunk):
  def __init__(self, data_list, x, y, bsize, sess):
    Trunk.__init__(self, data_list, x, y, bsize, sess)

  def read(self, key, idx):
    img = cv2.imread(self.data_list[key][idx][0], 0)
    if img.size != 64 * 64:
      x_ = None
    else:
      x_ = np.array(img).astype(np.float32).reshape(64, 64, 1)
    y_ = np.array(self.data_list[key][idx][1]).astype(np.int64)
    return x_, y_

def model(H, x, training = True):
  reuse = None if training else True

  net = conv2d(x, 64, [3, 3], activation_fn = tf.nn.relu)
  net = conv2d(net, 64, [3, 3], activation_fn = tf.nn.relu)
  net = max_pool2d(net, [2, 2], padding = 'VALID')

  net = conv2d(net, 128, [3, 3], activation_fn = tf.nn.relu)
  net = conv2d(net, 128, [3, 3], activation_fn = tf.nn.relu)
  net = max_pool2d(net, [2, 2], padding = 'VALID')

  net = conv2d(net, 256, [3, 3], activation_fn = tf.nn.relu)
  net = max_pool2d(net, [2, 2], padding = 'VALID')

  net = conv2d(net, 256, [3, 3], activation_fn = tf.nn.relu)
  net = max_pool2d(net, [2, 2], padding = 'VALID')

  net = conv2d(net, 512, [3, 3], activation_fn = tf.nn.relu)
  net = max_pool2d(net, [2, 2], padding = 'VALID')

  ksize = net.get_shape().as_list()
  net = max_pool2d(net, [ksize[1], ksize[2]])
  net = fully_connected(flatten(net), 1024, activation_fn = tf.nn.relu)
  net = dropout(net, 0.5, is_training = training)
  logits = fully_connected(net, 2, activation_fn = tf.nn.relu)
  pred = tf.argmax(logits, axis = 1)
  return logits, pred

def build(args, dat, training, sess):
  with open(args.hype) as f:
    H = json.load(f)
    H['subset'] = args.subset
    H['save_dir'] = OUTPUT_DIR + 'subset' + str(H['subset'])
    if args.gpu != None:
      H['gpu'] = args.gpu
  with open(META_DIR + 'fp.json') as fpj:
    meta = json.load(fpj)

  logging.basicConfig(filename = 'fp_detect_' + str(H['subset']) + '.log',
        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)

  bsize = H['batch_size']
  x = tf.placeholder(tf.float32, shape = [64, 64, 1])
  y = tf.placeholder(tf.int64, shape = [])

  fptrunk = FPTrunk(dat, x, y, bsize, sess)
  Xt, Yt = tf.train.batch(fptrunk.q['train'].dequeue(), batch_size = bsize, capacity = bsize * 2)
  Xv, Yv = tf.train.batch(fptrunk.q['valid'].dequeue(), batch_size = bsize, capacity = bsize * 2)

  logits, pred = model(H, Xt, training)
  varst = tf.trainable_variables()
  gstep = tf.Variable(0, trainable = False)
  loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
      logits=logits, labels=tf.one_hot(indices = Yt, depth = 2)))
  FN = tf.reduce_sum(tf.cast(tf.not_equal(pred, Yt) & tf.equal(Yt, 1), tf.float32))
  FP = tf.reduce_sum(tf.cast(tf.not_equal(pred, Yt) & tf.equal(Yt, 0), tf.float32))
  acc = tf.reduce_sum(tf.cast(tf.equal(pred, Yt), tf.float32)) / tf.size(Yt, out_type = tf.float32)

  vFN = tf.reduce_sum(tf.cast(tf.not_equal(pred, Yt) & tf.equal(Yt, 1), tf.float32))
  vFP = tf.reduce_sum(tf.cast(tf.not_equal(pred, Yt) & tf.equal(Yt, 0), tf.float32))
  vAcc = tf.reduce_sum(tf.cast(tf.equal(pred, Yt), tf.float32)) / tf.size(Yt, out_type = tf.float32)
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
  train_opt = opt.apply_gradients([(capped[i], vars[i]) for i in range(len(vars))],
        global_step = gstep)

  tf.summary.scalar('loss', loss)
  tf.summary.scalar('FN', FN)
  tf.summary.scalar('FP', FP)
  tf.summary.scalar('acc', acc)
  tf.summary.scalar('vFN', vFN)
  tf.summary.scalar('vFP', vFP)
  tf.summary.scalar('vacc', vAcc)
  tf.summary.scalar('norm', tf.reduce_sum(norm))
  summary_op = tf.summary.merge_all()

  saver = tf.train.Saver(max_to_keep = None)
  writer = tf.summary.FileWriter(logdir = H['save_dir'], flush_secs = 10)
  return (H, x, y, Xt, Yt, Xv, Yv,
    logits, pred, varst, gstep, loss, FN, FP, acc, vFN, vFP, vAcc, opt,
    grads_vars, grads, vars, capped, norm, train_opt, summary_op, saver, writer, fptrunk)

def train(args):
  with open(args.hype) as f:
    H = json.load(f)
    H['subset'] = args.subset
    H['save_dir'] = OUTPUT_DIR + 'subset' + str(H['subset'])
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
    (H, x, y, Xt, Yt, Xv, Yv,
    logits, pred, varst, gstep, loss, FN, FP, acc, vFN, vFP, vAcc, opt,
    grads_vars, grads, vars, capped, norm, train_opt, summary_op, saver, writer, fptrunk) = build(args, dat, True, sess)

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
      tol_loss, tol_tfn, tol_tfp, tol_vfn, tol_vfp, tol_acc, tol_vacc = [0.0] * 7
      for step in range(1, train_batches):
        curX, curY = sess.run([Xt, Yt])
        _, tloss, tacc, fn, fp = sess.run([train_opt, loss, acc, FN, FP], feed_dict = {Xt: curX, Yt: curY})

        if step % 100 == 0:
          logging.info('Training batchs %d, avg loss %f, acc %f, FN %d/%d, FP %d/%d.' %
            (step, tol_loss / step, tol_acc / step, tol_tfn, step * bsize, tol_tfp, step * bsize))
          summary_str = sess.run(summary_op)
          writer.add_summary(summary_str, global_step = gstep.eval())
        tol_loss += tloss
        tol_tfn += fn
        tol_tfp += fp
        tol_acc += tacc

      for step in range(valid_batches):
        curX, curY = sess.run([Xv, Yv])
        vfn, vfp, vacc = sess.run([vFN, vFP, vAcc], feed_dict = {Xt : curX, Yt: curY})
        tol_vfn += vfn
        tol_vfp += vfp
        tol_vacc += vacc

      t = time.time() - tst
      print((epoch, t, tol_loss / (train_batches * bsize),
        float(tol_vfn) / (valid_batches * bsize), float(tol_vfp) / (valid_batches * bsize),
        tol_vacc / valid_batches))
      logging.info(('epoch %d, time elapse %f, training loss %f,' +
        ' valid avg FN %f, FP %f, acc %f.') % (epoch, t, tol_loss / (train_batches * bsize),
        float(tol_vfn) / (valid_batches * bsize), float(tol_vfp) / (valid_batches * bsize),
        tol_vacc / valid_batches))
      saver.save(sess, OUTPUT_DIR + 'subset' + str(H['subset']) + '/save.ckpt', global_step = gstep)

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
  if args.hype == None or args.subset == None:
    raise 'Error: Config file and subset need to be specified.'
  if args.train == 1:
    train(args)

if __name__ == '__main__':
  main()