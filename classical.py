# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2017-01-25 16:05:50
# @Last Modified by:   yancz1989
# @Last Modified time: 2017-01-25 22:02:41

import json
from env import *
import sklearn
import cv2
import numpy as np
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

def get_sample(tag):
  subset = 'subset' + str(tag)
  with open(META_DIR + subset + '-scan.json') as f:
    data_all = json.load(f)
  with open(META_DIR + subset + '.json') as f:
    jspos = json.load(f)
  positives = [anno['image_path'] for anno in jspos]
  samples = dict([(anno['image_path'], {
    'dat' : cv2.imread(SAMPLE_DIR + anno['image_path']).flatten(),
    'label' : 0
    }) for anno in data_all])
  for key in positives:
    samples[key[:-7] + 'scan.bmp']['label'] = 1
  return samples

def accuracy(Y_hat, Y):
  Y_hat = Y_hat.astype(int)
  Y = Y.astype(int)
  acc = np.sum(Y_hat == Y).astype(float) / len(Y)
  TN = (Y_hat ^ Y) & (Y == 0)
  FP = (Y_hat ^ Y) & (Y == 1)
  return [acc, TP, TN, FN, FP]

def main(H):
  np.random.seed(2012310818)
  samples = get_sample(H['subset'])
  X = np.array([samples[key]['dat'] for key in samples])
  Y = np.array([samples[key]['label'] for key in samples])
  idx = np.arange(len(X)).astype(int)
  np.random.shuffle(idx)
  train_idx = idx[0 : int(len(idx) * 0.8)]
  val_idx = idx[int(len(idx) * 0.8) : len(idx)]
  weight = float(np.sum(Y == 0)) / len(Y)
  weights = {0 : weight, 1 : 1 - weight}
  if H['model'] == 'logistic':
    model = LogisticRegression(penalty = H['penalty'], C = H['lambda'],
      class_weight = weights, tol  = H['tol'], n_jobs = H['jobs'], 
      verbose = H['verbose'])
  elif H['model'] == 'svm':
    model = SVC(C = H['lambda'], kernel = H['kernel'],
      tol = H['tol'], class_weight = weights, verbose = H['verbose'])
  elif H['model'] == 'gbdt':
    model = GradientBoostingClassifier(learning_rate = H['lr'], 
      max_depth = H['depth'], verbose = H['verbose'])
  print(H)
  print('start training...')
  model.fit(X[train_idx], Y[train_idx])
  Y_hat = model.predict(X[val_idx, :])
  print(accuracy(Y_hat, Y))

if __name__ == '__main__':
  with open(sys.argv[1]) as f:
    H = json.load(f)
    main(H)
  