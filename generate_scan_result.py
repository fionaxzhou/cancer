import data_util as util
import numpy as np
import os
import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import trans
import sys
import SimpleITK as sitk 

DATA_ROOT="/home/yangxm/Workspace/data/LUNA16/"
INPUT_FILE="/home/yangxm/Workspace/data/LUNA16/result/result.json"
OUTPUT_FILE="./test.out"

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

result_map = get_image_map(DATA_ROOT, INPUT_FILE, 0.01)
generate_result(result_map, OUTPUT_FILE)

