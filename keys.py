# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2017-01-19 14:56:37
# @Last Modified by:   yancz1989
# @Last Modified time: 2017-01-19 14:59:58

from utilities import readCSV
import sys

if __name__ == '__main__':
  lines = readCSV(sys.argv[1])
  keys = []
  for line in lines[1:]:
    keys.append(line[0])
  keys = list(set(keys))
  keys.sort()
  with open('test/keys.csv', "w") as f:
    for key in keys:
      f.write(key + "\n")