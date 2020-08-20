# -*- coding: utf-8 -*-
# @Time    : 2020/8/19 21:00
# @Author  : zxl
# @FileName: test.py

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score,f1_score,recall_score,precision_score,accuracy_score




x = tf.constant([[0.5,0.3,0.8],
                 [0.2,0.9,0.1],
                 [0.7,0.4,0.7]])
# _, indices = tf.math.top_k(x, k=1)
# row_idx = tf.expand_dims(tf.range(tf.shape(x)[0]),axis = 1)
# indices = tf.concat([row_idx,indices],axis = 1)
# result = tf.scatter_nd(indices = indices, updates=tf.ones_like(tf.squeeze(indices)), shape=tf.shape(x))

y = tf.reshape(tf.reduce_max(x,axis = 1),[-1,1])
z = x/y

with tf.Session() as sess:
    print(sess.run(y))
    print(sess.run(z))
    # print(sess.run(result))



# indices = tf.constant([[4], [3], [1], [7]])
# updates = tf.constant([9, 10, 11, 12])
# shape = tf.constant([8])
# scatter = tf.scatter_nd(indices, updates, shape)
# with tf.Session() as sess:
#   print(sess.run(scatter))
