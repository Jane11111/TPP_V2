# -*- coding: utf-8 -*-
# @Time    : 2020/8/19 21:00
# @Author  : zxl
# @FileName: test.py

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score,f1_score,recall_score,precision_score,accuracy_score
from Model.Modules.net_utils import gather_indexes


tensor = tf.constant([[1,1,1],
                      [2,2,2]])
res =tf.reduce_sum(tensor,axis = 1)
with tf.Session() as sess:
    print(sess.run(res))

    print('---------------')

    # print(sess.run(result))



# indices = tf.constant([[4], [3], [1], [7]])
# updates = tf.constant([9, 10, 11, 12])
# shape = tf.constant([8])
# scatter = tf.scatter_nd(indices, updates, shape)
# with tf.Session() as sess:
#   print(sess.run(scatter))
