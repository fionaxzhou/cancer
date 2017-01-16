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

def filterBoxes(boxes, threshold):
    filtered_boxes = []
    for box in boxes:
        if box[4] >= threshold:
            filtered_boxes.append(box)
    return filtered_boxes

def generate_result(data_root, input_file, threshold, output_file):
    with open(input_file) as fin:
        result_list = json.load(fin)
    with open(output_file) as fout:
        fout.write("seriesuid,coordX,coordY,coordZ,probability\n")
        for it in result_list:
            key, subset, z = parse_image_file(it['file'])
            src_file = os.path.join(
                data_root, subset, key + ".mhd")
            boxes = filterBoxes[it['box']]
            itkimage = sitk.ReadImage(filename)
            for box in boxes:
                world_box = voxel_2_world(
                    [z, box[1], box[0]], itkimage)
                csv_line = key + "," + str(world_box[2]) + "," + str(world_box[1]) + "," + str(world_box[0]) + "," + str(box[4])
                fout.write(csv_line + "\n")




