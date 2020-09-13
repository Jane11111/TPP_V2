# -*- coding: utf-8 -*-
# @Time    : 2020/9/12 9:52
# @Author  : zxl
# @FileName: sahp_self_attention.py

import tensorflow as tf

class SahpSelfAttention():


    def __init__(self):
        pass


    def normalize(self,inputs,
                  epsilon=1e-8,
                  scope="ln",
                  reuse=None):
        '''Applies layer normalization.

        Args:
          inputs: A tensor with 2 or more dimensions, where the first dimension has
          `batch_size`.
          epsilon: A floating number. A very small number for preventing ZeroDivision Error.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
          by the same name.

        Returns:
          A tensor with the same shape and data dtype as `inputs`.
        '''
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
            outputs = gamma * normalized + beta

        return outputs

    def multistack_multihead_self_attention(self,X,M,Mv,L,N,head_num,stack_num,dropout_rate, reuse=None, scope = 'multistack_attention'):

        with tf.variable_scope(scope,reuse=reuse):
            for stack in range(stack_num):
                X = self.multihead_self_attention(X=X,
                                                  M = M,
                                                  Mv = Mv,
                                                  L = L,
                                                  N = N,
                                                  head_num=head_num,
                                                  dropout_rate=dropout_rate,
                                                  reuse=reuse,
                                                  scope = 'block'+str(stack)+'_multihead_attention')
                return X




    def multihead_self_attention(self,X,M,Mv,L,N,head_num,dropout_rate,reuse=None,scope = 'multihead_self_attention'):
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
            for cur_head in range(head_num):

                att = tf.exp(tf.matmul(X,X,transpose_b=True)) # batch_size, L, L

                Wv = tf.get_variable(str(cur_head)+'_Wv',shape = (M,Mv))
                V = tf.reshape(tf.matmul(tf.reshape(X,[-1,M]),Wv),shape = [-1,L,Mv])

                # TODO mask the attention value
                masked_idx = tf.range(start = 1, limit = L+1,delta = 1) # L, L
                masks = tf.expand_dims(tf.sequence_mask(masked_idx,L),axis = 0)
                masks = tf.tile(masks,[N,1,1])
                paddings = tf.ones_like(att) * (-2 ** 32 +1)
                masked_att = tf.where(masks,att,paddings)

                #
                # masked_att = tf.exp(masked_att)
                # sum_masked_att = tf.reduce_sum(masked_att, axis = 2, keep_dims=True) # batch_size, L, 1
                # masked_att = masked_att/(sum_masked_att +1e-9)
                masked_att = tf.nn.softmax(masked_att)

                # Dropouts
                masked_att = tf.layers.dropout(masked_att,rate = dropout_rate)

                # Weighted sum
                Si = tf.matmul(masked_att,V) # batch_size, L, Mv
                S.append(Si)

            # Restore shape
            S = tf.concat(S,axis=2) # batch_size,L, head_num * Mv

            Wo = tf.get_variable('Wo',shape = (head_num * Mv, M))
            S = tf.reshape(tf.matmul(tf.reshape(S,[-1,head_num * Mv]),Wo), shape = [-1,L, M]) # batch_size, L, M

        # Residual connection
        S += X

        # normalize
        S = self.normalize(inputs = S)

        return S

