# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2017-01-11 09:10:31
# @Last Modified by:   yancz1989
# @Last Modified time: 2017-01-23 01:34:37

import tensorflow as tf
import matplotlib.pyplot as plt
import os
import json
import subprocess
import argparse
import shutil
import sys
import time; 
from PIL import Image
from train import build_forward
from utils import train_utils
from utils.annolist import AnnotationLib as al
from utils.stitch_wrapper import stitch_rects
from utils.train_utils import add_rectangles
from utils.rect import Rect
from utils.stitch_wrapper import stitch_rects
from evaluate import add_rectangles
import cv2
import SimpleITK as sitk
from cv2 import imread, imwrite
from utilities import trans, readImageMap, readFileNameMap, readResultMap, load_itk_image, normalizePlanes, worldToVoxelCoord, voxel_2_world, parse_image_file, filterBoxes, readCSV
from env import *

def get_image_map(data_root, result_list, threshold):
  result_map = {}
  for it in result_list:
    key, subset, z = parse_image_file(it['file'])
    src_file = os.path.join(
      data_root, subset, key + ".mhd")
    boxes = filterBoxes(it['box'], threshold)
    if not result_map.get(src_file):
      result_map[src_file] = []
    result_map[src_file].append((key, z, boxes))

  return result_map

def generate_result(data_root, result_list, output_file, thr = 0.1):
  result_map = get_image_map(data_root, result_list, thr)
  with open(output_file, 'w') as fout:
    fout.write("seriesuid,coordX,coordY,coordZ,probability\n")
    for fkey, val in result_map.items():
      itkimage = sitk.ReadImage(str(fkey))
      for it in val:
        key, z, boxes = it
        for box in boxes:
          world_box = voxel_2_world(
            [z, box[1], box[0]], itkimage)
          csv_line = key + "," + str(world_box[2]) + "," + str(world_box[1]) + "," + str(world_box[0]) + "," + str(box[4])
          print(csv_line)
          fout.write(csv_line + "\n")

# def generate_result(data_root, result_list, output_file, thr = 0.1):
#   with open(output_file, 'w') as fout:
#     fout.write("seriesuid,coordX,coordY,coordZ,probability\n")
#     for it in result_list:
#       key, subset, z = parse_image_file(it['file'])
#       src_file = str(data_root + subset + '/' + key + ".mhd")
#       itkimage = sitk.ReadImage(src_file)
#       boxes = filterBoxes(it['box'], thr)
#       for box in boxes:
#         world_box = voxel_2_world([z, box[1], box[0]], itkimage)
#         csv_line = (key + "," + str(world_box[2]) + "," + 
#             str(world_box[1]) + "," + str(world_box[0]) + "," + str(box[4]))
#         print(csv_line)
#         fout.write(csv_line + "\n")

def evaluate(H, valids, param_path, thr = 0.7, l = 60000, r = 120010, sep = 10000, with_anno = True):
  true_annos = al.parse(valids)
  L = range(l, r, sep)
  for iteration in L:
    tf.reset_default_graph()
    # print(H['batch_size'])
    x_in = tf.placeholder(tf.float32, name='x_in', shape=[H['image_height'], H['image_width'], 3])
    if H['use_rezoom']:
      pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas = build_forward(
      H, tf.expand_dims(x_in, 0), 'test', reuse=None)
      grid_area = H['grid_height'] * H['grid_width']
      pred_confidences = tf.reshape(tf.nn.softmax(
      tf.reshape(pred_confs_deltas, [grid_area * H['rnn_len'], 2])), [grid_area, H['rnn_len'], 2])
      if H['reregress']:
        pred_boxes = pred_boxes + pred_boxes_deltas
    else:
      pred_boxes, pred_logits, pred_confidences = build_forward(H, tf.expand_dims(x_in, 0), 'test', reuse=None)
    saver = tf.train.Saver()
    output_dir = OUTPUT_DIR + H['subset'] + '/'
    gpu_options = tf.GPUOptions()
    gpu_options.allow_growth=True
    config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config = config) as sess:
      sess.run(tf.initialize_all_variables())
      print('load from ' + (param_path + 'save.ckpt-%d' % iteration))
      saver.restore(sess, param_path + 'save.ckpt-%d' % iteration)

      annolist = al.AnnoList()
      rslt = []
      t = time.time()
      if not os.path.exists(output_dir + 'val'):
        os.makedirs(output_dir + 'val')
      for i in range(len(true_annos)):
        true_anno = true_annos[i]
        img = imread(SAMPLE_DIR + true_anno.imageName)
        feed = {x_in: img}
        (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes, pred_confidences], feed_dict=feed)
        pred_anno = al.Annotation()
        pred_anno.imageName = true_anno.imageName
        print(pred_anno.imageName)
        new_img, rects = add_rectangles(H, [img], np_pred_confidences, np_pred_boxes,
                        use_stitching=True, rnn_len=H['rnn_len'], min_conf=thr,
                        show_suppressed=False)
      
        pred_anno.rects = rects
        annolist.append(pred_anno)
        fname = true_anno.imageName
        if with_anno:
          imwrite(output_dir + 'val/' + fname[fname.rindex('/') + 1 : -4] + '_' + str(iteration) + '_pred.jpg', new_img)
          shutil.copy(SAMPLE_DIR + true_anno.imageName[:-4] + '_gt.bmp',
            output_dir + 'val/' + fname[fname.rindex('/') + 1 : -4] + '_gt.bmp')
        box_confs = trans(np_pred_boxes, H, np_pred_confidences, thr)
        ret = {
          'file' : fname,
          'box' : box_confs.tolist()
          } 
        rslt.append(ret)

      avg_time = (time.time() - t) / (i + 1)
      print('%f images/sec' % (1. / avg_time))
      # with open(output_dir + 'result_' + str(iteration) + '.json', 'w') as f:
      #   json.dump(rslt, f)
      # batch_gen(output_dir + 'csv_' + str(iteration) + '.csv', output_dir + 'result_' + str(iteration) + '.json')
      generate_result(TRUNK_DIR, rslt, output_dir + 'csv_' + str(iteration) + '.csv')

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--gpu', default=0)
  parser.add_argument('--logdir', default='data/output')
  parser.add_argument('--tau', default=0.25, type=float)
  parser.add_argument('--subset', default=None, type=str)
  parser.add_argument('--left', default=60000, type=int)
  parser.add_argument('--right', default=100000, type=int)
  parser.add_argument('--sep', default=10000, type=int)
  parser.add_argument('--thr', default=0.5, type=float)
  args = parser.parse_args()
  os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
  output = 'subset' + args.subset
  hypes_file = '%s/hypes.json' % (OUTPUT_DIR + output)
  with open(hypes_file, 'r') as f:
    H = json.load(f)
    H['subset'] = 'subset' + args.subset
    evaluate(H, META_DIR + output + '-scan.json', OUTPUT_DIR + output + '/', args.thr,
        args.left, args.right, args.sep, with_anno = False)

if __name__ == '__main__':
  main()
