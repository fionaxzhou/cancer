import inception_v1
import inception_v2
import inception_v3
import inception_v4
import tensorflow.contrib.slim as slim

def model(x, H, reuse, is_training=True):
  if H['inception'] == 1:
    with slim.arg_scope(inception_v1.inception_v1_arg_scope()):
      _, T = inception_v1.inception_v1(x,
                      is_training = is_training,
                      num_classes = 1001,
                      dropout_keep_prob = 0.8,
                      spatial_squeeze = False,
                      reuse=reuse)
    coarse_feat = T['Mixed_5b']

    # fine feat can be used to reinspect input
    attention_lname = H.get('attention_lname', 'Mixed_3b')
    early_feat = T[attention_lname]
    early_feat_channels = 480
  elif H['inception'] == 3:
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
      _, T = inception_v3.inception_v3(x,
                      is_training = is_training,
                      num_classes = 1001,
                      dropout_keep_prob = 0.8,
                      spatial_squeeze = False,
                      reuse=reuse)
    coarse_feat = T['Mixed_5b']

    # fine feat can be used to reinspect input
    attention_lname = H.get('attention_lname', 'Mixed_3b')
    early_feat = T[attention_lname]
    early_feat_channels = 480

  return coarse_feat, early_feat, early_feat_channels
