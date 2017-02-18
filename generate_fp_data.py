import utilities as util
import numpy as np
import os
import json
from PIL import Image, ImageDraw
from env import *

MOV_LIST = [(-10, -10), (-10, 0), (-10, 10), (0, -10),
            (0, 10), (10, -10), (10, 0), (10, 10)]

def readImageMap(filename):
  lines = util.readCSV(filename)
  result = {}
  for line in lines[1:]:
    worldCoord = np.asarray(
      [float(line[3]), float(line[2]), float(line[1])])
    label = int(line[4])
    if not result.get(line[0]):
      result[line[0]] = []
    result[line[0]].append((worldCoord, label))
  return result

def generate_data(data_root, data_map, fp_dir):
  list_dirs = os.walk(data_root)
  index = 0
  for i in range(10):
    util.mkdir(FP_DIR + 'subset' + str(i))
  meta = dict([('subset' + str(i), []) for i in range(10)])
  for root, dirs, files in list_dirs:
    for f in files:
      if f.lower().endswith("mhd"):
        print(f)
        key = os.path.splitext(f)[0]
        subset = root.split("/")[-1]
        if key in data_map:
          numpyImage, numpyOrigin, numpySpacing = util.load_itk_image(os.path.join(root, f))
          for it in data_map[key]:
            worldCoord, label = it
            voxelCoord = util.worldToVoxelCoord(worldCoord, numpyOrigin, numpySpacing)
            voxelWidth = 65
            x = voxelCoord[1]
            y = voxelCoord[2]
            z = int(voxelCoord[0])
            patch = numpyImage[z, x - voxelWidth / 2:x + voxelWidth / 2,
                                  y - voxelWidth / 2:y + voxelWidth / 2]
            patch = util.normalizePlanes(patch)
            if patch.size == 0:
              continue

            fpath = os.path.join(fp_dir, subset + '/patch_' + str(index)  + '.bmp')
            Image.fromarray(patch * 255).convert('L').save(fpath)
            meta[subset].append((fpath, label))
            index += 1

            if label == 1:
              for i in range(50):
                dx, dy = MOV_LIST[i % 8]
                xx = x + int(dx * np.random.rand())
                yy = y + int(dy * np.random.rand())
                aug_patch = numpyImage[z, xx - voxelWidth / 2:xx + voxelWidth / 2,
                                       yy - voxelWidth / 2:yy + voxelWidth / 2]
                aug_patch = util.normalizePlanes(aug_patch)
                if aug_patch.size == 0:
                  continue
                fpath = os.path.join(fp_dir, subset + '/patch_' + str(index)  + '.bmp')
                Image.fromarray(aug_patch * 255).convert('L').save(fpath)
                meta[subset].append((fpath, label))
                index += 1
  with open(META_DIR + 'fp.json', 'w') as f:
    json.dump(meta, f)

data_map = readImageMap(CANDIDATE_CSV)
generate_data(TRUNK_DIR, data_map, FP_DIR)

