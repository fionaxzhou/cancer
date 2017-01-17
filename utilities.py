import SimpleITK as sitk
import numpy as np
import csv
import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import SimpleITK as sitk
from cv2 import imread, imwrite

def load_itk_image(filename):
  itkimage = sitk.ReadImage(filename)
  numpyImage = sitk.GetArrayFromImage(itkimage)
  numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
  numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
  return numpyImage, numpyOrigin, numpySpacing


def readCSV(filename):
  lines = []
  with open(filename, "rb") as f:
    csvreader = csv.reader(f)
    for line in csvreader:
      lines.append(line)
  return lines

def voxel_2_world(voxel_coord, itkimage):
  world_coord = list(reversed(
    itkimage.TransformContinuousIndexToPhysicalPoint(list(reversed(voxel_coord)))))
  return world_coord

def voxelCoordToWorld(voxelCoord, origin, spacing):
  stretchedVoxelCoord = voxelCoord * spacing
  worldCoord = stretchedVoxelCoord + origin
  return worldCoord

def worldToVoxelCoord(worldCoord, origin, spacing):
  stretchedVoxelCoord = np.absolute(worldCoord - origin)
  voxelCoord = stretchedVoxelCoord / spacing
  return voxelCoord


def normalizePlanes(npzarray):
  maxHU = 400.
  minHU = -1000.
  npzarray = (npzarray - minHU) / (maxHU - minHU)
  npzarray[npzarray > 1] = 1.
  npzarray[npzarray < 0] = 0.
  return npzarray

def readFileNameMap(map_filename):
  file_map = {}
  with open(map_filename) as map_file:
    file_name_list = json.load(map_file)
  for it in file_name_list:
    file_map[it['ID_name']] = it['long_name']
  return file_map

def parse_image_file(filename):
  cols = filename.split("-")
  subset = cols[0]
  key = cols[1]
  z_axis = int(cols[2])
  return key, subset[:subset.index('/')], z_axis

def filterBoxes(boxes, threshold):
  filtered_boxes = []
  for box in boxes:
    if box[4] >= threshold:
      filtered_boxes.append(box)
  return filtered_boxes

def readResultMap(result_filename, file_map, threshold):
  result_map = {}
  with open(result_filename) as result_file:
    result_list = json.load(result_file)
  for it in result_list:
    filename = it['file']
    key = file_map[filename]
    key = os.path.splitext(key)[0]
    boxes = it['box']
    boxes = filterBoxes(boxes, threshold)
    if not result_map.get(key):
      result_map[key] = []
    cols = filename.split('_')
    index = int(cols[2])
    result_map[key].append((index, boxes))
  for key, val in result_map.iteritems():
    val.sort()
  return result_map

def readImageMap(filename):
  lines = readCSV(filename)
  result = {}
  for line in lines[1:]:
    worldCoord = np.asarray(
      [float(line[3]), float(line[2]), float(line[1])])
    radius = float(line[4]) / 2.0 + 1.0 
    if not result.get(line[0]):
      result[line[0]] = []
    result[line[0]].append((worldCoord, radius))
  return result


def trans(boxes, H, confs, thr = -1.0):
  gw = H['grid_width']
  gh = H['grid_height']
  cell_pix_size = H['region_size']
  rnnl = H['rnn_len']
  ncls = H['num_classes']
  boxes = np.reshape(boxes, (-1, gh, gw, rnnl, 4))
  confs = np.reshape(confs, (-1, gh, gw, rnnl, ncls))
  ret = []
  for i in range(rnnl):
    for y in range(gh):
      for x in range(gw):
        if np.max(confs[0, y, x, i, 1:]) > thr:
          box = boxes[0, y, x, i, :]
          abs_cx = int(box[0]) + cell_pix_size/2 + cell_pix_size * x
          abs_cy = int(box[1]) + cell_pix_size/2 + cell_pix_size * y
          w = box[2]
          h = box[3]
          ret.append([abs_cx, abs_cy, w, h, np.max(confs[0, y, x, i, 1: ])])
  return np.array(ret)


def split(meta_root, samples):
  np.random.seed(2012310818)
  l = len(samples)    
  idxes = np.random.permutation(np.arange(l))
  train = [dat[i] for i in idxes[0 : int(l * 0.7)]]
  vals = [dat[i] for i in idxes[int(l * 0.7) : ]]

  with open(meta_root + 'train.json', 'w') as g:
    json.dump(train, g)
  with open(meta_root + 'vals.json', 'w') as g:
    json.dump(vals, g)


def writeCSV(filename, lines):
  with open(filename, "wb") as f:
    csvwriter = csv.writer(f)
    csvwriter.writerows(lines)

def tryFloat(value):
  try:
    value = float(value)
  except:
    value = value
  
  return value

def getColumn(lines, columnid, elementType=''):
  column = []
  for line in lines:
    try:
      value = line[columnid]
    except:
      continue
        
    if elementType == 'float':
      value = tryFloat(value)

    column.append(value)
  return column

