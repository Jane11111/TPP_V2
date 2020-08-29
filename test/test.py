# -*- coding: utf-8 -*-
# @Time    : 2020/8/28 16:09
# @Author  : zxl
# @FileName: test.py

import tensorflow as tf

from Model.Modules.net_utils import gather_indexes, layer_norm



# demo 取出上上一个时间的时间

# batch_size = 2
# seq_len = 3
#
a = tf.constant([[1,2,3],
                 [4,5,6]])

col_idx = tf.reshape(tf.constant([0,1]),[-1,1])
row_idx = tf.reshape(tf.range(start = 0, limit = a.shape[0],delta = 1),[-1,1])
idx = tf.concat([row_idx,col_idx],axis = 1)

res = tf.gather_nd(a,idx)



with tf.Session() as sess:
    print(sess.run(res))