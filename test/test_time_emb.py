# -*- coding: utf-8 -*-
# @Time    : 2020/8/19 21:00
# @Author  : zxl
# @FileName: test.py

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score,f1_score,recall_score,precision_score,accuracy_score
from Model.Modules.net_utils import gather_indexes


batch_size = 2
seq_len = 3
M = 5


time_lst = tf.constant([[0,0,0],
                        [1,2,2]],dtype=tf.float32)

single_odd_mask =  np.zeros(shape = (M,))
single_odd_mask[::2] = 1
single_odd_mask = tf.convert_to_tensor(single_odd_mask,dtype=tf.float32) # M,
single_even_mask = np.zeros(shape = (M,))
single_even_mask[1::2] = 1
single_even_mask = tf.convert_to_tensor(single_even_mask,dtype=tf.float32)

emb_time_lst = tf.tile(tf.expand_dims(time_lst,axis = 2),[1,1,M]) # batch_size, seq_len, M

single_odd_deno = tf.to_float(10000 ** (tf.range(start = 0, limit = M, delta = 1)/M))# M,
single_even_deno = tf.to_float(10000 **(tf.range(start = 1, limit = M+1, delta = 1)/M))

odd_emb = tf.cos(emb_time_lst/single_odd_deno)
even_emb = tf.sin(emb_time_lst/single_even_deno)
time_lst_emb = odd_emb * single_odd_mask + even_emb * single_even_mask

# b = tf.Variable(tf.random_normal([1]))
# loss = tf.reduce_mean(tf.reduce_mean(masked_emb,axis = 2),axis = 1) + b
# train = tf.train.AdamOptimizer(0.001).minimize(loss)


with tf.Session() as sess:
    print(sess.run(time_lst_emb))
    # print(sess.run(even_mask))
    print('---------------')


    # cur_time_lst = np.array([[0., 0., 0.],
    #                         [1., 2., 2.]])
    # feed_dict = {}
    # feed_dict[time_lst] = cur_time_lst
    # sess.run(tf.global_variables_initializer(), feed_dict)
    # sess.run(tf.local_variables_initializer(), feed_dict)
    #
    # for epoch in range(10):
    #     cur_time_lst = np.random.rand(2,3)
    #     feed_dict = {}
    #     feed_dict[time_lst] = cur_time_lst
    #     sess.run(train, feed_dict )
    #     print(sess.run(loss,feed_dict))


    # print(sess.run(result))



# indices = tf.constant([[4], [3], [1], [7]])
# updates = tf.constant([9, 10, 11, 12])
# shape = tf.constant([8])
# scatter = tf.scatter_nd(indices, updates, shape)
# with tf.Session() as sess:
#   print(sess.run(scatter))
