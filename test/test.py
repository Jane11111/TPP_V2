# -*- coding: utf-8 -*-
# @Time    : 2020/8/28 16:09
# @Author  : zxl
# @FileName: test.py

import numpy as np
import tensorflow as tf

from Model.Modules.net_utils import gather_indexes, layer_norm

inputs = tf.random_normal(shape=(3,4))

res=inputs.get_shape().with_rank(2)[1]
with tf.Session() as sess:
    print(sess.run(res))