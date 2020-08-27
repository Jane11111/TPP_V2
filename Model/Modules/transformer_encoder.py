# -*- coding: utf-8 -*-
# @Time    : 2020/8/27 19:26
# @Author  : zxl
# @FileName: transformer_encoder.py

import tensorflow as tf

class transformer_encoder():


    def __init__(self):
        pass


    def stack_multihead_self_attention(self,stack_num,X,M,Mk,Mv,head_num,reuse= None,scope = 'stack_multihead_self_attention'):

        with tf.get_variable(scope = scope, reuse= reuse):

            for i in range(stack_num):
                X = self.multihead_self_attention(X = X,
                                                  M = M,
                                                  Mk = Mk,
                                                  Mv = Mv,
                                                  head_num= head_num,
                                                  reuse=reuse,
                                                  scope = 'block_'+str(i)+'_multihead_self_attention')
        return X # batch_size, L, M




    def multihead_self_attention(self,X,M,Mk,Mv,head_num,reuse=None,scope = 'multihead_self_attention'):
        """

        :param X: batch_size, shape = (batch_size,L, M)
        :param M: M
        :param Mk: Mk
        :param Mv: Mv
        :param head_num: int
        :param reuse:
        :param scope:
        :return:
        """

        with tf.variable_scope(scope,reuse=reuse):
            S = []
            L= X.shape[1]
            for cur_head in range(head_num):
                Wq = tf.get_variable(str(cur_head)+'_Wq',shape = (M,Mk))
                Wk = tf.get_variable(str(cur_head)+'_Wk',shape = (M,Mk))
                Wv = tf.get_variable(str(cur_head)+'_Wv',shape = (M,Mv))

                Q = tf.reshape(tf.matmul(tf.reshape(X,[-1,M]),Wq),shape = [-1,L,Mk])# batch_size, L, Mk
                K = tf.reshape(tf.matmul(tf.reshape(X,[-1,M]),Wk),shape = [-1,L,Mk])
                V = tf.reshape(tf.matmul(tf.reshape(X,[-1,M]),Wv),shape = [-1,L,Mk])

                att = tf.matmul(Q,K,transpose_b=True)/(Mk**0.5) # batch_size, L, L
                # TODO mask the attention value
                masked_idx = tf.range(start = 1, limit = L+1,delta = 1)
                masks = tf.expand_dims(tf.sequence_mask(masked_idx,L))
                masks = tf.tile(masks,[X.shape[0],1,1])
                paddings = tf.ones_like(att) * (-1 ** 32 +1)
                masked_att = tf.where(masks,att,paddings)

                Si = tf.matmul(tf.nn.softmax(masked_att),V) # batch_size, L, Mv
                S.append(Si)

            Wo = tf.get_variable('Wo',shape = (head_num * Mv, M))
            S = tf.concat(S,axis=2) # batch_size,L, head_num * Mv

            S = tf.reshape(tf.matmul(tf.reshape(S,[-1,head_num * Mv]),Wo), shape = [-1,L, M]) # batch_size, L, M
        return S

