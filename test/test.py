# -*- coding: utf-8 -*-
# @Time    : 2020/8/28 16:09
# @Author  : zxl
# @FileName: test.py

import numpy as np
import tensorflow as tf
from Model.Modules.net_utils import gather_indexes

batch_size = 2
sims_len = 3
time_lst = tf.constant([[1,2,3],[4,5,0]])
seq_len = tf.constant([3,2])
mask_index = tf.expand_dims(seq_len-1, axis=1)
col_idx =  mask_index - 1
row_idx = tf.reshape(tf.range(start=0, limit= batch_size, delta=1), [-1, 1])
idx = tf.concat([row_idx, col_idx], axis=1)
last_time = tf.gather_nd( time_lst, idx)
with tf.Session() as sess:
    print(sess.run(last_time))


    # print(type(arr1[0]))

    # print(sess.run(cur_mu))