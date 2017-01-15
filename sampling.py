import data_util as util
import numpy as np
import os
import os.path
import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from utilities import readCSV, load_itk_image, worldToVoxelCoord, readImageMap, split
from env import *


def readImageMap(filename):
  lines = util.readCSV(filename)
  result = {} 
  for line in lines[1:]:
    worldCoord = np.asarray(
      [float(line[3]), float(line[2]), float(line[1])])
    radius = float(line[4]) / 2.0 + 1.0 
    if not result.get(line[0]):
      result[line[0]] = []
    result[line[0]].append((worldCoord, radius))
  return result


def generateFullImage(data_root, data_map, output_dir, prefix, box = False):
  list_dirs = os.walk(data_root)
  result = []
  mapping =  []
  index = 0
  samples = dict([('subset' + str(i), []) for i in range(10)])
  for root, dirs, files in list_dirs:
    for f in files:
      if f.lower().endswith('mhd'):
        key = os.path.splitext(f)[0]
        subset = root[root.rindex('/') + 1: ]
        print(subset, key)
        if key in data_map:
          numpyImage, numpyOrigin, numpySpacing = util.load_itk_image(
            os.path.join(root, f))

          for it in data_map[key]:
            worldCoord, radius = it
            voxelCoord = util.worldToVoxelCoord(
              worldCoord, numpyOrigin, numpySpacing)
            voxelWidth = 65
            patch = numpyImage[voxelCoord[0], 0:512, 0:512]
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
            img_name = subset + '/luna_label_%d_512x512.bmp' % index
            im.save(os.path.join(output_dir, img_name))
            # if box:
            draw = ImageDraw.Draw(im)
            draw.line([(x1, y1), (x1, y2)], fill=255, width=2)
            draw.line([(x1, y1), (x2, y1)], fill=255, width=2)
            draw.line([(x1, y2), (x2, y2)], fill=255, width=2)
            draw.line([(x2, y1), (x2, y2)], fill=255, width=2)
            del draw
            im.save(os.path.join(output_dir, img_name[:-4] + '_gt.bmp'))
            index += 1

            cur = {
              "image_path": img_name,
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
            mapping.append({'ID_name' : img_name, 'long_name' : f})

  for key in samples:
    with open(META_DIR + key + '.json', 'w') as f:
      json.dump(samples[key], f)

  with open(META_DIR + 'mapping.json', 'w') as f:
    json.dump(mapping, f)

def generateImagePatch(data_root, data_map, output_dir, prefix):
  list_dirs = os.walk(data_root)
  index = 0
  for root, dirs, files in list_dirs:
    for f in files:
      if f.lower().endswith('mhd'):
        key = os.path.splitext(f)[0]
        if key in data_map:
          numpyImage, numpyOrigin, numpySpacing = util.load_itk_image(os.path.join(root, f))
          worldCoord = data_map[key]
          voxelCoord = util.worldToVoxelCoord(worldCoord, numpyOrigin, numpySpacing)
          voxelWidth = 65
          patch = numpyImage[voxelCoord[0], voxelCoord[1] - voxelWidth / 2:voxelCoord[1] + voxelWidth / 2, voxelCoord[2] - voxelWidth / 2:voxelCoord[2] + voxelWidth / 2]
          patch = util.normalizePlanes(patch)
          Image.fromarray(patch * 255).convert('L').save(os.path.join(output_dir, 'patch_' + prefix + '_' +  str(index)  + '.jpg'))
          index += 1

if __name__ == '__main__':
  dirs = [SAMPLE_DIR, META_DIR, OUTPUT_DIR,] + [SAMPLE_DIR + 'subset' + str(i) for i in range(10)]
  for d in dirs:
    if not os.path.exists(d):
      os.mkdir(d)

  positive_map = readImageMap(ANNOTATION_CSV)
  generateFullImage(TRUNK_DIR, positive_map, SAMPLE_DIR, "rect")
