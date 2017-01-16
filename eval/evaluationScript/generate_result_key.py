import csv
import data_util as util
import sys


def get_result_keys(filename):
    lines = util.readCSV(filename)
    keys = []
    for line in lines[1:]:
        keys.append(line[0])
    keys = list(set(keys))
    keys.sort()
    return keys


def main(argv):
    result_file = argv[0]
    output_file = argv[1]
    keys = get_result_keys(result_file)
    with open(output_file, "w") as f:
        for key in keys:
            f.write(key + "\n")

if __name__ == "__main__":
    main(sys.argv[1:])
