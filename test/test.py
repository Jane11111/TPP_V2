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
a = tf.constant([1,2,3])
b = tf.constant([0,1,0])
c = tf.reduce_sum(a)



with tf.Session() as sess:
    print(sess.run(c))