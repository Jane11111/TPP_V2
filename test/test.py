# -*- coding: utf-8 -*-
# @Time    : 2020/8/28 16:09
# @Author  : zxl
# @FileName: test.py

import numpy as np
import tensorflow as tf

a = tf.constant([1,2,3])
b = tf.squeeze(a)
c = tf.squeeze(b)

with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(c))
    print(a.shape)
    print(c.shape)

    # print(type(arr1[0]))

    # print(sess.run(cur_mu))