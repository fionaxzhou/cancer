import data_util as util
import numpy as np
import os
import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

DATA_ROOT = '/home/yangxm/Workspace/data/LUNA16/subset0'
OUTPUT_DIR = '/home/yangxm/Workspace/data/LUNA16/scan/subset0'

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

generate_scan_image(DATA_ROOT, OUTPUT_DIR)

