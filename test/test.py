# -*- coding: utf-8 -*-
# @Time    : 2020/8/28 16:09
# @Author  : zxl
# @FileName: test.py

import numpy as np
import tensorflow as tf
from Model.Modules.net_utils import gather_indexes

a = tf.constant([[0.,1]])
b = tf.constant([[3.,3.]])

tmp = tf.cast(tf.to_int32(a>0.) + tf.to_int32(a<0.),dtype=tf.bool)
c = tf.where(tmp,a, b)
with tf.Session() as sess:
    print(sess.run(tmp))
    print(sess.run(b))
    print(sess.run(c))

    # print(sess.run(d))
    # print(type(arr1[0]))

    # print(sess.run(cur_mu))