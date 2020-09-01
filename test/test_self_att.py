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

time_enc = tf.random_normal(shape = (batch_size, seq_len, M))
type_enc = tf.random_normal(shape = (batch_size, seq_len, M))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.initialize_all_variables())
    transformer_model = transformer_encoder()


    stack_num = 1
    head_num = 1

    new_X = transformer_model.stack_multihead_self_attention(stack_num=stack_num,
                                                             type_enc=type_enc,
                                                             time_enc=time_enc,
                                                             M = M,
                                                             Mi = 1,
                                                             Mk = Mk,
                                                             Mv = Mv,
                                                             dropout_rate=0.1,
                                                             L = seq_len,
                                                             N = batch_size,
                                                             head_num=head_num)
    print(sess.run(new_X))
