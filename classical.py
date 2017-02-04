# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2017-01-25 16:05:50
# @Last Modified by:   yancz1989
# @Last Modified time: 2017-02-03 09:24:30

import json
from env import *
import sklearn
import cv2
import numpy as np
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier

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

def sample_gen(tag, batch_size):
  subset = 'subset' + str(tag)
  with open(META_DIR + subset + '-scan.json') as f:
    data_all = json.load(f)
  with open(META_DIR + subset + '.json') as f:
    jspos = json.load(f)
  positives = [anno['image_path'] for anno in jspos]
  negatives = [anno[]]
  samples = dict([(anno['image_path'], {
    'dat' : None,
    'label' : 0
    }) for anno in data_all])
  for key in positives:
    samples[key[:-7] + 'scan.bmp']['label'] = 1

  idx = np.arange(len(samples)).astype(int)
  np.random.shuffle(idx)
  for batch_idx in range(0, len(idx), batch_size):

  return samples

def accuracy(Y_hat, Y):
  Y_hat = Y_hat.astype(int)
  Y = Y.astype(int)
  acc = np.sum(Y_hat == Y).astype(float) / len(Y)

  TN = (Y_hat ^ Y) & (Y == 0)
  FP = (Y_hat ^ Y) & (Y == 1)
  return [acc, sum(TN), len(Y), sum(FP), sum(Y_hat != Y), sum(Y)]

def main(H, load = False):
  np.random.seed(2012310818)
  # weight = float(np.sum(Y == 0)) / len(Y)
  # weights = {0 : weight, 1 : 1 - weight}
  if load == False:
    if H['model'] == 'logistic':
      # model = LogisticRegression(penalty = H['penalty'], C = H['lambda'],
      #   class_weight = weights, tol  = H['tol'], n_jobs = H['jobs'], 
      #   verbose = H['verbose'])
      model = SGDClassifier(loss = 'log', penalty = H['penalty'], alpha = 1.0 / H['lambda'])
        # class_weight = weights, )
    elif H['model'] == 'svm':
      # model = SVC(C = H['lambda'], kernel = H['kernel'],
      #   tol = H['tol'], class_weight = weights, verbose = H['verbose'])
      model = SGDClassifier(loss = 'hinge', penalty = H['penalty'], alpha = 1.0 / H['lambda'])
    elif H['model'] == 'gbdt':
      model = GradientBoostingClassifier(learning_rate = H['lr'], 
        max_depth = H['depth'], verbose = H['verbose'])
    print('start training...')
    if H['sgd'] == False:
      samples = get_sample(H['subset'])
      X = np.array([samples[key]['dat'] for key in samples])
      Y = np.array([samples[key]['label'] for key in samples])
      idx = np.arange(len(X)).astype(int)
      np.random.shuffle(idx)
      train_idx = idx[0 : int(len(idx) * 0.8)]
      val_idx = idx[int(len(idx) * 0.8) : len(idx)]
      model.fit(X[train_idx], Y[train_idx])
    else:

    with open(str(H['subset']) + '_' + H['model'] + '.pkl', 'w') as f:
      joblib.dump(model, f)
  else:
    with open(str(H['subset']) + '_' + H['model'] + '.pkl') as f:
      model = joblib.load(f)
  Y_hat = model.predict(X[val_idx, :])
  print(accuracy(Y_hat, Y[val_idx]))

if __name__ == '__main__':
  with open(sys.argv[1]) as f:
    H = json.load(f)
    print(H)
    main(H, True)
  