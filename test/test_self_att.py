# -*- coding: utf-8 -*-
# @Time    : 2020/8/28 14:14
# @Author  : zxl
# @FileName: test.py


import tensorflow as tf
from Model.Modules.transformer_encoder import transformer_encoder

batch_size = 2
seq_len = 3
M = 4
Mk = 3
Mv = 5

X = tf.random_normal(shape = (batch_size, seq_len, M))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.initialize_all_variables())
    transformer_model = transformer_encoder()


    stack_num = 1
    head_num = 1

    new_X = transformer_model.stack_multihead_self_attention(stack_num=stack_num,
                                                             X = X,
                                                             M = M,
                                                             Mk = Mk,
                                                             Mv = Mv,
                                                             head_num=head_num)
    print(sess.run(new_X))
