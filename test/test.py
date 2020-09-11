# -*- coding: utf-8 -*-
# @Time    : 2020/8/28 16:09
# @Author  : zxl
# @FileName: test.py

import numpy as np
import tensorflow as tf

from Model.Modules.net_utils import gather_indexes, layer_norm

now_batch_size = 3
time_lst = tf.constant([[1,2,3,4],
                         [5,6,7,8],
                         [9,10,0,0]])
mask_index = tf.constant([[3],
                          [3],
                          [1]])


col_idx = mask_index - 1
row_idx = tf.reshape(tf.range(start=0, limit=now_batch_size, delta=1), [-1, 1])
idx = tf.concat([row_idx, col_idx], axis=1)
last_time = tf.gather_nd(time_lst, idx)
with tf.Session() as sess:
    print(sess.run(last_time))