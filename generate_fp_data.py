import data_util as util
import numpy as np
import os
import json
from PIL import Image, ImageDraw

DATA_ROOT = '/home/yangxm/Workspace/data/LUNA16/'
CANDIDATE_CSV = ('/home/yangxm/Workspace/data/LUNA16/CSVFILES/candidates.csv')
POSITIVE_DIR = ('/home/yangxm/Workspace/data/LUNA16/classification/fp/positive/')
NEGATIVE_DIR = ('/home/yangxm/Workspace/data/LUNA16/classification/fp/negative/')

MOV_LIST = [(-10, -10), (-10, 0), (-10, 10), (0, -10), (0, 10), (10, -10), (10, 0), (10, 10)]

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

def generate_data(data_root, data_map, positive_dir, negative_dir, aug_data):
    list_dirs = os.walk(data_root)
    index = 0
    for root, dirs, files in list_dirs:
        for f in files:
            if f.lower().endswith("mhd"):
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
                        patch = numpyImage[z, x - voxelWidth / 2:x + voxelWidth / 2, y - voxelWidth / 2:y + voxelWidth / 2]
                        patch = util.normalizePlanes(patch)
                        if label == 1:
                            Image.fromarray(patch * 255).convert('L').save(os.path.join(positive_dir, 'patch_' + str(index)  + '.jpg'))
                            index += 1
                            if aug_data:
                                for xy in MOV_LIST:
                                    dx, dy = xy
                                    xx = x + dx
                                    yy = y + dy
                                    aug_patch = numpyImage[z, xx - voxelWidth / 2:xx + voxelWidth / 2, yy - voxelWidth / 2:yy + voxelWidth / 2]
                                    aug_patch = util.normalizePlanes(aug_patch)
                                    Image.fromarray(aug_patch * 255).convert('L').save(os.path.join(positive_dir, 'patch_' + str(index)  + '.jpg'))
                                    index += 1



                        else:
                            Image.fromarray(patch * 255).convert('L').save(os.path.join(negative_dir, 'patch_' + str(index)  + '.jpg'))
                            index += 1

data_map = readImageMap(CANDIDATE_CSV)
generate_data(DATA_ROOT, data_map, POSITIVE_DIR, NEGATIVE_DIR, True)

