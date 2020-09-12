# -*- coding: utf-8 -*-
# @Time    : 2020/8/28 16:09
# @Author  : zxl
# @FileName: test.py

import numpy as np
import tensorflow as tf

from Model.Modules.net_utils import gather_indexes, layer_norm
from Model.Modules.net_utils import gather_indexes


batch_size = 2
L = 5
num_units = 4
a = tf.constant([[1,1,2,3,0],
                 [2,3,4,0,0]])
seq_len = tf.constant([4,3])
mask_index = tf.expand_dims(seq_len-1, axis = 1)
res = tf.squeeze(gather_indexes(batch_size=batch_size,
                     seq_length=L,
                     width = 1,
                     sequence_tensor=tf.expand_dims(a,axis = -1),
                     positions = mask_index-1))
with tf.Session() as sess:
    print(sess.run(res))