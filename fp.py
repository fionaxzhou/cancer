# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2017-02-03 16:18:59
# @Last Modified by:   yancz1989
# @Last Modified time: 2017-02-04 17:48:53

import tensorflow as tf
import numpy as np
from env import *
import argparse
import os
import tensorflow.contrib.slim as slim
import sys
import time
import cv2
import threading
import json
import logging
import utils.inception_v2
from tensorflow.contrib.layers import convolution2d as conv2d
from tensorflow.contrib.layers import max_pool2d, fully_connected, avg_pool2d, dropout, flatten
from tensorflow.contrib.layers import xavier_initializer_conv2d as init_conv
from tensorflow.contrib.layers import variance_scaling_initializer as init_fc


def build(H, x, training = True):
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

def train(args):
  logging.basicConfig(filename = 'running.log', 
        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)
  with open(args.hype) as f:
    H = json.load(f)
    H['subset'] = args.subset
    H['save_dir'] = OUTPUT_DIR + 'subset' + str(H['subset'])
  with open(META_DIR + 'fp.json') as fpj:
    meta = json.load(fpj)

  bsize = H['batch_size']
  x = tf.placeholder(tf.float32, shape = [64, 64, 1])
  y = tf.placeholder(tf.int64, shape = [])

  q, eq_op= {}, {}
  for training in ['train', 'valid']:
    q[training] = tf.FIFOQueue(capacity = bsize * 4, dtypes = [tf.float32, tf.int64],
        shapes = ([64, 64, 1], []))
    eq_op[training] = q[training].enqueue([x, y])
  Xt, Yt = tf.train.batch(q['train'].dequeue(), batch_size = bsize, capacity = bsize * 2)
  Xv, Yv = tf.train.batch(q['valid'].dequeue(), batch_size = bsize, capacity = bsize * 2)

  dat = {}
  dat['train'] = []
  dat['valid'] = []
  for i in range(10):
    if i == H['subset']:
      dat['valid'] = meta['subset' + str(i)]
    else:
      dat['train'] += meta['subset' + str(i)]
  train_batches = len(dat['train']) / bsize
  valid_batches = len(dat['valid']) / bsize

  def data_generator(sess, dataset, key = 'train'):
    dat = dataset[key]
    idx = np.arange(len(dat))
    while True:
      np.random.shuffle(idx)
      for i in range(len(idx)):
        img = cv2.imread(dat[idx[i]][0], 0)
        if img.size != 64 * 64:
          continue
        x_ = np.array(img).astype(np.float32).reshape(64, 64, 1)
        y_ = np.array(dat[idx[i]][1]).astype(np.int64)
        sess.run(eq_op[key], feed_dict = {x : x_, y : y_})

  logits, pred = build(H, Xt, True)
  varst = tf.trainable_variables()
  gstep = tf.Variable(0, trainable = False)
  loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
      logits, tf.one_hot(indices = Yt, depth = 2)))
  TN = tf.reduce_sum(tf.cast(tf.not_equal(pred, Yt) & tf.equal(Yt, 1), tf.float32))
  FP = tf.reduce_sum(tf.cast(tf.not_equal(pred, Yt) & tf.equal(Yt, 0), tf.float32))
  acc = tf.reduce_sum(tf.cast(tf.equal(pred, Yt), tf.float32)) / tf.size(Yt, out_type = tf.float32)

  vTN = tf.reduce_sum(tf.cast(tf.not_equal(pred, Yt) & tf.equal(Yt, 1), tf.float32))
  vFP = tf.reduce_sum(tf.cast(tf.not_equal(pred, Yt) & tf.equal(Yt, 0), tf.float32))
  vacc = tf.reduce_sum(tf.cast(tf.equal(pred, Yt), tf.float32)) / tf.size(Yt, out_type = tf.float32)
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
  tf.summary.scalar('TN', TN)
  tf.summary.scalar('FP', FP)
  tf.summary.scalar('acc', acc)
  tf.summary.scalar('vTN', vTN)
  tf.summary.scalar('vFP', vFP)
  tf.summary.scalar('vacc', vacc)
  tf.summary.scalar('norm', tf.reduce_sum(norm))


  os.environ['CUDA_VISIBLE_DEVICES'] = str(H['gpu'])
  gpu_options = tf.GPUOptions()
  gpu_options.allow_growth=True
  config = tf.ConfigProto(gpu_options=gpu_options)

  saver = tf.train.Saver(max_to_keep = None)
  writer = tf.train.SummaryWriter(logdir = H['save_dir'], flush_secs = 10)

  with tf.Session(config = config) as sess:
    sess.run(tf.global_variables_initializer())
    eq_threads = {}
    for key in ['train', 'valid']:
      eq_threads[key] = threading.Thread(target = data_generator, args = [sess, dat, key])
      eq_threads[key].isDaemon()
      eq_threads[key].start()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord, sess = sess)
    if args.weight != None:
      logging.info('Restoring from %s...' % args.weight)
      saver.restore(sess, weight)

    for epoch in range(H['epochs']):
      tst = time.time()
      tol_loss, tol_ttn, tol_tfp, tol_vtn, tol_vfp, tol_acc, tol_vacc = [0.0] * 7
      for step in range(1, train_batches + 1):
        curX, curY = sess.run([Xt, Yt])
        _, tloss, tacc, tn, fp = sess.run([train_opt, loss, acc, TN, FP], feed_dict = {Xt: curX, Yt: curY})
        if step % 100 == 0:
          logging.info('Training batchs %d, avg loss %f, acc %f, TN %d/%d, FP %d/%d.' % 
            (step, tol_loss / step, tol_acc / step, tol_ttn, step * bsize, tol_tfp, step * bsize))
        tol_loss += tloss
        tol_ttn += tn
        tol_tfp += fp
        tol_acc += tacc

      for step in range(valid_batches):
        curX, curY = sess.run([Xv, Yv])
        vtn, vfp, vacc = sess.run([vTN, vFP, vacc], feed_dict = {Xt : curX, Yt: curY})
        tol_vtn += vtn
        tol_vfp += vfp
        tol_vacc += vacc

      logging.info('')
      t = time.time() - tst
      logging.info(('epoch %d, time elapse %f, training loss %f,',
        ' valid avg TN %d/%d, %d/%d, acc %f.') % (epoch, t, tol_loss / (train_batches * bsize),
        float(tol_vtn) / (valid_batches * bsize), float(tol_vfp) / (valid_batches * bsize)))
      saver.save(sess, OUTPUT_DIR + 'subset' + H['subset'] + '/save.ckpt')

    logging.info('training finished, try ending...')
    sess.run(q.close(cancel_pending_enqueues=True))
    coord.request_stop()
    coord.join(threads)
    sess.close()
    logging.info('ended...')

def validate():
  pass

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--hype', default = None, type = str)
  parser.add_argument('--weight', default = None, type = str)
  parser.add_argument('--subset', default = None, type = int)
  parser.add_argument('--train', default = 1, type = int)
  args = parser.parse_args()
  if args.hype == None or args.subset == None:
    raise 'Error: Config file and subset need to be specified.'
  tf.set_random_seed(2012310818)
  if args.train == 1:
    train(args)
  else:
    validate(args)

if __name__ == '__main__':
  main()