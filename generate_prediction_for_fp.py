import data_util as util
import numpy as np
import os
import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import trans
import sys
import SimpleITK as sitk 

def voxel_2_world(voxel_coord, itkimage):
    world_coord = list(reversed(
        itkimage.TransformContinuousIndexToPhysicalPoint(list(reversed(voxel_coord)))))
    return world_coord

def parse_image_file(filename):
    cols = filename.split("-")
    subset = cols[0]
    key = cols[1]
    z_axis = int(cols[2])
    return key, subset, z_axis

def filterBoxes(boxes, threshold, output_dir):
    filtered_boxes = []
    for box in boxes:
        if box[4] >= threshold:
            filtered_boxes.append(box)
    return filtered_boxes

def generate_result(data_root, input_file, threshold, output_dir):
    with open(input_file) as fin:
        result_list = json.load(fin)
    for it in result_list:
        key, subset, z = parse_image_file(it['file'])
        filename = os.path.join(data_root, subset, key + ".mhd")
        boxes = filterBoxes[it['box']]
        numpyImage, numpyOrigin, numpySpacing = util.load_itk_image(filename)
        voxelWidth = 65
        prefix = subset + '-' + key + '-' + str(z) + '-'
        index = 0
        for box in boxes:
            x = box[0]
            y = box[1]
            patch = numpyImage[z, x - voxelWidth / 2:x + voxelWidth / 2, y - voxelWidth / 2:y + voxelWidth / 2]
            Image.fromarray(patch * 255).convert('L').save(os.path.join(output_dir,  prefix + str(index)  + '.jpg'))
            index = index + 1




