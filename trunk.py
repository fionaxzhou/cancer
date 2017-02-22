# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2017-02-14 14:49:16
# @Last Modified by:   yancz1989
# @Last Modified time: 2017-02-21 22:10:46

import tensorflow as tf
import threading
import numpy as np

class Trunk(object):
  # parameters:
  # data_list: a list containing train and valid, inner structure of list is not
  #     specified, which you can implement it via "read" interface.
  # x: input variable, a tensorflow placeholder of any shapes and types.
  # y: output variable, aka the label, of any shapes and types.
  # bsize: batch size of input
  def __init__(self, data_list, x, y, bsize, sess):
    self.data_list = data_list
    self.x = x
    self.y = y
    self.bsize = bsize
    self.sess = sess
    self.q, self.eq_op, self.X, self.Y, self.nbatches, self.eq_threads = {}, {}, {}, {}, {}, {}
    for key in ['train', 'valid']:
      self.q[key] = tf.FIFOQueue(capacity = self.bsize * 2,
        dtypes = [self.x.dtype, self.y.dtype], shapes = (x.get_shape(), y.get_shape()))
      self.eq_op[key] = self.q[key].enqueue([self.x, self.y])
      self.X[key], self.Y[key] = tf.train.batch(self.q[key].dequeue(),
        batch_size = self.bsize, capacity = self.bsize * 2)
      self.nbatches[key] = len(data_list[key]) / self.bsize

  def data_generator(self, key = 'train'):
    dat = self.data_list[key]
    idx = np.arange(len(dat))
    while True:
      np.random.shuffle(idx)
      for i in range(len(idx)):
        x_, y_ = self.read(key, idx[i])
        if x_ is None or y_ is None:
          continue
        self.sess.run(self.eq_op[key], feed_dict = {self.x : x_, self.y : y_})

  def start(self):
    for key in ['train', 'valid']:
      self.eq_threads[key] = threading.Thread(target = self.data_generator, args = [key])
      self.eq_threads[key].isDaemon()
      self.eq_threads[key].start()
    self.coord = tf.train.Coordinator()
    self.threads = tf.train.start_queue_runners(coord = self.coord, sess = self.sess)

  def read(self, key, idx):
    raise NotImplementedError('Error! one need to implement its own read interface')

  def stop(self):
    logging.info('training finished, try ending...')
    for key in ['train', 'valid']:
      self.sess.run(self.q[key].close(cancel_pending_enqueues=True))
    self.coord.request_stop()
    self.coord.join(threads)

  def get_batch(self, key = 'train'):
    curX, curY = sess.run([X[key], Y[key]])
    return curX, curY

