# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2017-01-17 23:43:18
# @Last Modified by:   yancz1989
# @Last Modified time: 2017-01-17 23:46:52
import utilities as util
import numpy as np
import os
import sys
from PIL import Image, ImageDraw

def generate_scan_image(data_root, output_dir):
  list_dirs = os.walk(data_root)
  for root, dirs, files in list_dirs:
    for f in files:
      if f.lower().endswith('mhd'):
        key = os.path.splitext(f)[0]
        subset = root.split("/")[-1]
        numpyImage, numpyOrigin, numpySpacing = (
          util.load_itk_image(
            os.path.join(root, f)))
        for z in range(numpyImage.shape[0]):
          patch = numpyImage[z, 0:512, 0:512]
          patch = util.normalizePlanes(patch)
          im = Image.fromarray(patch * 255).convert('L')
          output_filename = (
            subset + "-" + key + "-" + str(z) + "-scan.bmp")
          im.save(os.path.join(
            output_dir, output_filename))

if __name__ == '__main__':
  generate_scan_image(sys.argv[1], sys.argv[2])