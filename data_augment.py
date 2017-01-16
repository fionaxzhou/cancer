from PIL import Image, ImageDraw
import copy
import os
import json
import sys

SIZE = 512

def draw_box(im, box):
    draw = ImageDraw.Draw(im)
    x1 = box["x1"]
    y1 = box["y1"]
    x2 = box["x2"]
    y2 = box["y2"]
    draw.line([(x1, y1), (x1, y2)], fill=255, width=2)
    draw.line([(x1, y1), (x2, y1)], fill=255, width=2)
    draw.line([(x1, y2), (x2, y2)], fill=255, width=2)
    draw.line([(x2, y1), (x2, y2)], fill=255, width=2)
    del draw
    return im


def read_source_json(filename):
    with open(filename) as f:
        data_map = json.load(f)
    return data_map

def flip_image(input_file, input_box):
    prefix = os.path.splitext(input_file)[0]
    x1 = input_box[0]["x1"]
    y1 = input_box[0]["y1"]
    x2 = input_box[0]["x2"]
    y2 = input_box[0]["y2"]
    im = Image.open(input_file)
    im1 = im.transpose(Image.FLIP_LEFT_RIGHT)
    im2 = im.transpose(Image.FLIP_TOP_BOTTOM)
    output_file = [
        prefix + "_flip_1.bmp", prefix + "_flip_2.bmp"]
    output_box = [
        {
            "x1": SIZE - 1 - x1,
            "x2": SIZE - 1 - x2,
            "y1": y1,
            "y2": y2,
        },
        {
            "x1": x1,
            "x2": x2,
            "y1": SIZE - 1 - y1,
            "y2": SIZE - 1 - y2,
        },
    ]

    im1.save(output_file[0])
    im2.save(output_file[1])
    # draw_box(im1, output_box[0]).save(output_file[0])
    # draw_box(im2, output_box[1]).save(output_file[1])
    return output_file, output_box

def rotate_image(input_file, input_box):
    prefix = os.path.splitext(input_file)[0]
    x1 = input_box[0]["x1"]
    y1 = input_box[0]["y1"]
    x2 = input_box[0]["x2"]
    y2 = input_box[0]["y2"]
    im = Image.open(input_file)
    im1 = im.transpose(Image.ROTATE_90)
    im2 = im.transpose(Image.ROTATE_180)
    im3 = im.transpose(Image.ROTATE_270)
    output_file = [
        prefix + "_rotate_1.bmp",
        prefix + "_rotate_2.bmp",
        prefix + "_rotate_3.bmp",
    ]
    output_box = [
        {
            "x1": y1,
            "x2": y2,
            "y1": SIZE - 1 - x1,
            "y2": SIZE - 1 - x2,
        },
        {
            "x1": SIZE - 1 - x1,
            "x2": SIZE - 1 - x2,
            "y1": SIZE - 1 - y1,
            "y2": SIZE - 1 - y2,
        },
        {
            "x1": SIZE - 1 - y1,
            "x2": SIZE - 1 - y2,
            "y1": x1,
            "y2": x2,
        },
    ]
    im1.save(output_file[0])
    im2.save(output_file[1])
    im3.save(output_file[2])
    # draw_box(im1, output_box[0]).save(output_file[0])
    # draw_box(im2, output_box[1]).save(output_file[1])
    # draw_box(im3, output_box[2]).save(output_file[2])

    return output_file, output_box

def main(argv):
    input_json = argv[0]
    output_json = argv[1]
    data_map = read_source_json(input_json)
    result_map = copy.deepcopy(data_map)
    for it in data_map:
        input_file = it["image_path"]
        input_box = it["rects"]
        flip_file, flip_box = flip_image(
            input_file, input_box)
        for index in range(len(flip_file)):
            item = {
                "image_path": flip_file[index],
                "rects": [flip_box[index]],
            }
            result_map.append(item)

        rotate_file, rotate_box = rotate_image(
            input_file, input_box)
        for index in range(len(rotate_file)):
            item = {
                "image_path": rotate_file[index],
                "rects": [rotate_box[index]],
            }
            result_map.append(item)
    print json.dumps(result_map, sort_keys=True)


if __name__ == "__main__":
    main(sys.argv[1:])
