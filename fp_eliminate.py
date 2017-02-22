# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2017-02-21 11:01:12
# @Last Modified by:   yancz1989
# @Last Modified time: 2017-02-22 00:25:01

import os
import numpy as np
import cv2
import logging
import json
import tensorflow as tf
import argparse

from env import *
from fp import model
from evaluate import get_image_map, generate_result
from utilities import mkdir

def eliminate(args):
  with open(args.hype) as f:
    H = json.load(f)
    H['subset'] = args.subset
    if args.gpu != None:
      H['gpu'] = args.gpu
    H['epoch'] = args.epoch
    H['weights'] = args.weights
    H['fpepoch'] = args.fpepoch
    H['save_dir'] = 'data/output.eliminate/' + 'subset' + str(H['subset'])
    mkdir(H['save_dir'])

  os.environ['CUDA_VISIBLE_DEVICES'] = str(H['gpu'])
  gpu_options = tf.GPUOptions()
  gpu_options.allow_growth = True
  config = tf.ConfigProto(gpu_options = gpu_options)
  tf.set_random_seed(2012310818)

  with tf.Session(config = config) as sess:
    xv = tf.placeholder(tf.float32, shape = [1, 64, 64, 1])
    logits, pred = model(H, xv, training = True)

    saver = tf.train.Saver(max_to_keep = None)
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, H['weights'])
    voxelW = 65
    SUBSET_DIR = DETECT_DIR + 'subset' + str(H['subset']) + '/'

    with open(SUBSET_DIR + 'result_' + str(H['epoch']) + '.json') as f:
      detects = json.load(f)
      i = 0
      for it in detects:
        boxes = it['box']
        if len(boxes) > 0:
          img = cv2.imread(SAMPLE_DIR + it['file'], 0).astype(np.float32)
          rboxes = []
          for box in boxes:
            x, y = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
            if x - voxelW // 2 < 0:
              x = 0
            if x + voxelW // 2 >= img.shape[1]:
              x = img.shape[1] - voxelW
            if y - voxelW // 2 < 0:
              y = 0
            if y + voxelW // 2 >= img.shape[0]:
              y = img.shape[0] - voxelW
            patch = img[y : y + voxelW - 1, x: x + voxelW - 1]

            y_, logits_ = sess.run([pred, logits], feed_dict = {xv : patch.reshape(1, 64, 64, 1)})
            if y_ == 1:
              rboxes.append(box)
              cv2.imwrite(str(i) + '.bmp', patch)
              i = i + 1
              print(logits_, y_)
          it['box'] = rboxes

    generate_result(TRUNK_DIR, detects, SUBSET_DIR + str(H['epoch']) + '.csv', 0.1)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--hype', default = 'hypes/fp.json', type = str)
  parser.add_argument('--weight', default = None, type = str)
  parser.add_argument('--subset', default = None, type = int)
  parser.add_argument('--gpu', default = None, type = int)
  parser.add_argument('--epoch', default = None, type = int)
  parser.add_argument('--weights', default = None, type = str)
  parser.add_argument('--fpepoch', default = 95447, type = int)
  args = parser.parse_args()
  logging.basicConfig(format='%(asctime)s %(message)s',
      datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)
  eliminate(args)

if __name__ == '__main__':
  main()

