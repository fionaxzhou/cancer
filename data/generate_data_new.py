import data_util as util
import numpy as np
import os
import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

DATA_ROOT = '/home/yangxm/Workspace/data/LUNA16/'
ANNOTATION_CSV = ('/home/yangxm/Workspace/data/LUNA16/CSVFILES/annotations.csv')
OUTPUT_DIR = ('/home/yangxm/Workspace/data/LUNA16'
              '/detection/rect2/')


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

def generate_data(data_root, data_map, output_dir, box = False):
    list_dirs = os.walk(data_root)
    result = []
    origin_map = {}
    spacing_map = {}
    for root, dirs, files in list_dirs:
        for f in files:
            if f.lower().endswith("mhd"):
                key = os.path.splitext(f)[0]
                subset = root.split("/")[-1]
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
                        if box:
                            draw = ImageDraw.Draw(im)
                            draw.line([(x1, y1), (x1, y2)], fill=255, width=2)
                            draw.line([(x1, y1), (x2, y1)], fill=255, width=2)
                            draw.line([(x1, y2), (x2, y2)], fill=255, width=2)
                            draw.line([(x2, y1), (x2, y2)], fill=255, width=2)
                            del draw
                        img_name = subset + "-" + key + "-" + str(z) + "-512.bmp"
                        im.save(os.path.join(output_dir, img_name))
                        cur = {
                            "image_path": "./data/" + img_name,
                            "rects": [
                                {
                                    "x1": x1,
                                    "x2": x2,
                                    "y1": y1,
                                    "y2": y2,
                                }
                            ]
                        }
                        result.append(cur)
    return result, origin_map, spacing_map

data_map = readImageMap(ANNOTATION_CSV)
result, origin_map, spacing_map = generate_data(DATA_ROOT, data_map, OUTPUT_DIR)
