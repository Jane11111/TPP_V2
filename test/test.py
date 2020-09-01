# -*- coding: utf-8 -*-
# @Time    : 2020/8/28 16:09
# @Author  : zxl
# @FileName: test.py

import numpy as np
import tensorflow as tf

from Model.Modules.net_utils import gather_indexes, layer_norm



# demo 取出上上一个时间的时间

# batch_size = 2
# seq_len = 3
#
L = 3
N = 2
att = ([[[0.1,0.2,0.7],
        [0.3,0.5,0.2],
        [0.9,0.0,0.1]],
        [[0.1, 0.2, 0.7],
         [0.3, 0.5, 0.2],
         [0.9, 0.0, 0.1]],
        ])
masked_idx = tf.range(start = 1, limit = L+1,delta = 1) # L, L
masks = tf.expand_dims(tf.sequence_mask(masked_idx,L),axis = 0)
masks = tf.tile(masks,[N,1,1])
paddings = tf.ones_like(att) * (-2 ** 32 +1)
masked_att = tf.where(masks,att,paddings)
masked_att = tf.nn.softmax(masked_att)
with tf.Session() as sess:
    print(sess.run(masked_att))


x1 = pow(np.e,0.1)
x2 = pow(np.e,0)
x3 = pow(np.e,0)
print(x2/(x1+x2+x3))
