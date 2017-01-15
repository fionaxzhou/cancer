# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2017-01-11 09:10:31
# @Last Modified by:   yancz1989
# @Last Modified time: 2017-01-15 11:54:12

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
from utilities import trans, readImageMap, readFileNameMap, readResultMap, load_itk_image, normalizePlanes, worldToVoxelCoord, voxel_2_world
from env import *

def generateCSV(csv_f, data_root, data_map, result_map, prefix):
  print 'seriesuid,coordX,coordY,coordZ,probability'
  list_dirs = os.walk(data_root)
  key_set = {}
  result = []
  img_index = 0
  for root, dirs, files in list_dirs:
    for f in files:
      if f.lower().endswith('mhd'):
        key = os.path.splitext(f)[0]

        if key in data_map and key in result_map and len(data_map[key]) == len(result_map[key]):

          filename = os.path.join(root, f)
          itkimage = sitk.ReadImage(filename)
          numpyImage, numpyOrigin, numpySpacing = (load_itk_image(filename))
          for index, value in  enumerate(data_map[key]):
            worldCoord, radius = value
            voxelCoord = worldToVoxelCoord(worldCoord, numpyOrigin, numpySpacing)
            voxelWidth = 65
            patch = numpyImage[voxelCoord[0], 0:512, 0:512]
            patch = normalizePlanes(patch)
            im = Image.fromarray(patch * 255).convert('L')

            _, boxes = result_map[key][index]
            for box in boxes:

              x1 = box[0] - box[2] / 2
              y1 = box[1] - box[3] / 2
              x2 = box[0] + box[2] / 2
              y2 = box[1] + box[3] / 2

              world_box = voxel_2_world(
                [voxelCoord[0], box[1], box[0]], itkimage)

              csv_line = key + "," + \
                str(world_box[2]) + "," + str(world_box[1]) + \
                "," + str(world_box[0]) + "," + str(box[4])
              print(csv_line)
              csv_f.write(csv_line)
              csv_f.write('\n')


def batch_gen(csv_file, result_file):
  data_map = readImageMap(ANNOTATION_CSV)
  file_map = readFileNameMap(MAP_FILE)
  result_map = readResultMap(result_file, file_map, 0.0001)
  with open(csv_file, 'w') as f:
    generateCSV(f, TRUNK_DIR, data_map, result_map, "rect")

def evaluate(H, valids, param_path):
  true_annos = al.parse(valids)

  L = range(30000, 60001, 5000)
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
    with tf.Session() as sess:
      sess.run(tf.initialize_all_variables())
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
        new_img, rects = add_rectangles(H, [img], np_pred_confidences, np_pred_boxes,
                        use_stitching=True, rnn_len=H['rnn_len'], min_conf=0.05,
                        show_suppressed=False)
      
        pred_anno.rects = rects
        annolist.append(pred_anno)
        fname = true_anno.imageName
        imwrite(output_dir + 'val/' + fname[fname.rindex('/') + 1 : -4] + '_' + str(iteration) + '_pred.jpg', new_img)
        shutil.copy(SAMPLE_DIR + true_anno.imageName[:-4] + '_gt.bmp',
          output_dir + 'val/' + fname[fname.rindex('/') + 1 : -4] + '_gt.bmp')
        box_confs = trans(np_pred_boxes, H, np_pred_confidences, 0.001)
        ret = {
          'file' : fname,
          'box' : box_confs.tolist()
          } 
        rslt.append(ret)

      avg_time = (time.time() - t) / (i + 1)
      print('%f images/sec' % (1. / avg_time))
      with open(output_dir + 'result_' + str(iteration) + '.json', 'w') as f:
        json.dump(rslt, f)
      batch_gen(output_dir + 'csv_' + str(iteration) + '.csv', output_dir + 'result_' + str(iteration) + '.json')

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--gpu', default=0)
  parser.add_argument('--logdir', default='output')
  parser.add_argument('--tau', default=0.25, type=float)
  parser.add_argument('--subset', default=None, type=str)
  args = parser.parse_args()
  os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
  output = 'subset' + args.subset
  hypes_file = '%s/hypes.json' % (OUTPUT_DIR + output)
  with open(hypes_file, 'r') as f:
    H = json.load(f)
    H['subset'] = 'subset' + args.subset
    evaluate(H, META_DIR + output + '.json', OUTPUT_DIR + output + '/')

if __name__ == '__main__':
  main()
