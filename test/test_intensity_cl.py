# -*- coding: utf-8 -*-
# @Time    : 2020/8/29 11:34
# @Author  : zxl
# @FileName: test_intensity_cl.py

import tensorflow as tf


from Model.Modules.intensity_calculation import thp_intensity_calculation




intensity_model = thp_intensity_calculation()

batch_size = 1
num_units = 64

type_num = 75

hidden_emb = tf.constant( [[ 0.29215693,  0.4489792 , -0.53350675 , 0.35743615, -0.30777854 ,-0.14418337,
   0.06991651 , 0.5605727,  -0.46704412  ,0.0512832  , 0.03191641  ,0.03121886,
  -0.75917995, -0.19405083  ,0.540873,   -0.35349736 , 0.18284392, -0.3571576,
   0.09077607 , 0.01643857 , 0.01223318  ,0.05231029, -0.15211263 , 0.33404177,
  -0.39161062 ,-0.3199696 , -0.48239124, -0.12737677, -0.9906788,   0.23922439,
   0.23855579, -0.00707929,  0.3561414,  -0.380953,    0.04258061, -0.22703983,
   0.74553895 ,-0.3137054,  -0.42700619 , 0.42869794  ,0.02354719 , 0.09466644,
  -0.33760184 , 0.13240297, -0.41858178, -0.19427028 ,-0.21381083,  0.0313789,
  -0.39215773 , 0.58935636, -0.18623005 , 0.4483724 , -0.49445438 , 0.30186978,
  -0.02922207,  1.0328494,  -0.14854024,  0.49852648 , 0.62923807,-0.49552146,
  -0.21122593 ,-0.7947048,  -0.1577854 ,  0.34499103]])
last_time = tf.constant([1e-9 ])
target_time = tf.constant([1.3269231])

target_intensity = intensity_model.cal_target_intensity(hidden_emb=hidden_emb,
                                                       target_time=target_time,
                                                       last_time= last_time,
                                                       type_num = type_num)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    print(sess.run(target_intensity))

