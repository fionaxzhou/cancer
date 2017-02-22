# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2017-01-17 23:43:18
# @Last Modified by:   yancz1989
# @Last Modified time: 2017-02-22 20:33:29
import utilities as util
from utilities import parse_image_file, filterBoxes, voxel_2_world, mkdir
import numpy as np
import os
import json
import sys
from PIL import Image, ImageDraw
import SimpleITK as sitk
from env import *

def generate_scan_image(subset):
  list_dirs = os.walk(TRUNK_DIR + subset)
  jsobjs = []
  output_dir = SAMPLE_DIR + subset
  mkdir(output_dir)
  for root, dirs, files in list_dirs:
    for f in files:
      if f.lower().endswith('mhd'):
        key = os.path.splitext(f)[0]
        numpyImage, numpyOrigin, numpySpacing = (
          util.load_itk_image(
            os.path.join(root, f)))
        for z in range(numpyImage.shape[0]):
          patch = numpyImage[z, 0:512, 0:512]
          patch = util.normalizePlanes(patch)
          im = Image.fromarray(patch * 255).convert('L')
          output_filename = (
            subset + "-" + key + "-" + str(z) + "-scan.bmp")
          print(subset + '/' + output_filename)
          im.save(os.path.join(
            output_dir, output_filename))
          jsobjs.append({
                "image_path": subset + '/' + output_filename,
                "rects":[]
              }
            )
  with open(META_DIR + subset + '-scan.json', 'w') as f:
    json.dump(jsobjs, f)


def get_image_map(data_root, input_file, threshold):
  result_map = {}
  with open(input_file) as f:
    result_list = json.load(f)
  for it in result_list:
    key, subset, z = parse_image_file(it['file'])
    src_file = os.path.join(
      data_root, subset, key + ".mhd")
    boxes = filterBoxes(it['box'], threshold)
    if not result_map.get(src_file):
      result_map[src_file] = []
    result_map[src_file].append((key, z, boxes))

  return result_map

def generate_result(result_map, output_file):
  with open(output_file) as fout:
    fout.write("seriesuid,coordX,coordY,coordZ,probability\n")
    for fkey, val in result_map.items():
      itkimage = sitk.ReadImage(fkey)
      for it in val:
        key, z, boxes = val
        for box in boxes:
          world_box = voxel_2_world(
            [z, box[1], box[0]], itkimage)
          csv_line = key + "," + str(world_box[2]) + "," + str(world_box[1]) + "," + str(world_box[0]) + "," + str(box[4])
        fout.write(csv_line + "\n")

if __name__ == '__main__':
  if sys.argv[1] == 'gen':
    generate_scan_image(sys.argv[2])
  else:
    result_map = get_image_map(TRUNK_DIR, sys.argv[2], 0.01)
    generate_result(result_map, OUTPUT_FILE)