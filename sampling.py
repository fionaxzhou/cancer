import utilities as util
import numpy as np
import os
import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from env import *
# from utilities import readImageMap, readResultMap, readFileNameMap, readCSV

def generate_data(data_root, data_map, output_dir, box = False):
  list_dirs = os.walk(data_root)
  samples = dict([('subset' + str(i), []) for i in range(10)])
  origin_map = {}
  spacing_map = {}
  for root, dirs, files in list_dirs:
    for f in files:
      if f.lower().endswith("mhd"):
        key = os.path.splitext(f)[0]
        subset = root.split("/")[-1]
        pimg = output_dir + subset
        if key in data_map:
          numpyImage, numpyOrigin, numpySpacing = util.load_itk_image(os.path.join(root, f))
          origin_map[key] = numpyOrigin
          spacing_map[key] = numpySpacing
          for it in data_map[key]:
            worldCoord, radius = it
            voxelCoord = util.worldToVoxelCoord(
              worldCoord, numpyOrigin, numpySpacing)
            voxelWidth = 65
            z = int(voxelCoord[0])
            patch = numpyImage[z, 0:512, 0:512]
            patch = util.normalizePlanes(patch)
            im = Image.fromarray(patch * 255).convert('L')

            p1 = [worldCoord[0], worldCoord[1] - radius, worldCoord[2] - radius]
            p2 = [worldCoord[0], worldCoord[1] + radius, worldCoord[2] + radius]
            q1 = util.worldToVoxelCoord(
              p1, numpyOrigin, numpySpacing)
            q2 = util.worldToVoxelCoord(
              p2, numpyOrigin, numpySpacing)
            y1 = q1[1] - 2
            y2 = q2[1] + 2
            x1 = q1[2] - 2
            x2 = q2[2] + 2
            img_name = subset + "-" + key + "-" + str(z) + "-512.bmp"
            im.save(os.path.join(pimg, img_name))
            draw = ImageDraw.Draw(im)
            draw.line([(x1, y1), (x1, y2)], fill=255, width=2)
            draw.line([(x1, y1), (x2, y1)], fill=255, width=2)
            draw.line([(x1, y2), (x2, y2)], fill=255, width=2)
            draw.line([(x2, y1), (x2, y2)], fill=255, width=2)
            del draw
            im.save(os.path.join(pimg, img_name[:-4] + '_gt.bmp'))
            cur = {
              "image_path": subset + '/' + img_name,
              "rects": [
                {
                  "x1": x1,
                  "x2": x2,
                  "y1": y1,
                  "y2": y2,
                }
              ]
            }
            samples[subset].append(cur)

  for key in samples:
    with open(META_DIR + key + '.json', 'w') as f:
      json.dump(samples[key], f)


if __name__ == '__main__':
  dirs = [SAMPLE_DIR, META_DIR, OUTPUT_DIR,] + [SAMPLE_DIR + 'subset' + str(i) for i in range(10)]
  for d in dirs:
    if not os.path.exists(d):
      os.mkdir(d)
  data_map = util.readImageMap(ANNOTATION_CSV)
  generate_data(TRUNK_DIR, data_map, SAMPLE_DIR)

# generate_scan_image(DATA_ROOT, OUTPUT_DIR)
