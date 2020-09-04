# -*- coding: utf-8 -*-
# @Time    : 2020-08-19 09:51
# @Author  : zxl
# @FileName: base_model.py



import os
import time
from util.model_log import create_log
from Model.Modules.net_utils import *


class base_model(object):


    def __init__(self,FLAGS,Embedding):
        self.FLAGS = FLAGS
        self.learning_rate = tf.placeholder(tf.float64,[],name = 'learning_rate') #learning_rate都设置成输入了
        self.llh_decay_rate = tf.placeholder(tf.float32, [], name='llh_decay_rate')
        if self.FLAGS.checkpoint_path_dir != None:
            self.checkpoint_path_dir = self.FLAGS.checkpoint_path_dir
        else:
            self.checkpoint_path_dir = 'data/check_point/' + self.FLAGS.model_name
        if not os.path.exists(self.checkpoint_path_dir):
            os.makedirs(self.checkpoint_path_dir)

        self.init_optimizer()
        self.embedding = Embedding
        log_ins = create_log()
        self.logger = log_ins.logger

    def init_variables(self, sess):

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())


    def init_optimizer(self):

        if self.FLAGS.optimizer == 'adadelta':
            self.opt = tf.train.AdadeltaOptimizer(learning_rate = self.learning_rate)
        elif self.FLAGS.optimizer == 'adam':
            self.opt = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        elif self.FLAGS.optimizer == 'rmsprop':
            self.opt = tf.train.RMSPropOptimizer(learning_rate= self.learning_rate)
        else:
            self.opt = tf.train.GradientDescentOptimizer(learning_rate= self.learning_rate)

    def build_model(self):
        pass


    def save(self, sess, global_step = None, path = None, variable_lst = None):

        if path == None:
            path = self.checkpoint_path_dir
        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(path, 'model.ckpt')

        saver = tf.train.Saver(var_list = variable_lst)
        save_path = saver.save(sess, save_path = path, global_step=global_step)


        self.logger.info('model saved at %s' % save_path)

    def restore(self, sess, path, variable_lst = None, graph_path = None):
        if graph_path != None:
            saver = tf.train.import_meta_graph(graph_path)
        else:
            saver = tf.train.Saver(var_list = variable_lst)

        saver.restore(sess, tf.train.latest_checkpoint(path))
        self.logger.info('model restored from %s' %path)
    def summary(self):

        self.merged = tf.summary.merge_all()

        time_array = time.localtime(time.time())
        time_str = time.strftime('%Y-%m-%d--%H-%M-%S', time_array)
        model_name = self.FLAGS.model_name
        filename = 'data/tensorboard_result/'

        filename = filename + model_name + '_' +self.FLAGS.loss+'_'+ time_str

        self.train_writer = tf.summary.FileWriter(filename + '/tensorboard_train')
        self.eval_writer = tf.summary.FileWriter(filename + '/tensorboard_eval')


    def cal_gradient(self,trainable_params):

        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _  = tf.clip_by_global_norm(gradients, self.FLAGS.max_gradient_norm)
        self.train_op = self.opt.apply_gradients(zip(clip_gradients, trainable_params))
        self.summary()

    def train(self, sess, batch_data, learning_rate,llh_decay_rate):

        input_dic = self.embedding.make_feed_dic(batch_data = batch_data)

        input_dic[self.learning_rate] = learning_rate
        input_dic[self.llh_decay_rate] = llh_decay_rate
        input_dic[self.now_batch_size] = len(batch_data)

        output_feed = [
                       self.loss,
                       self.log_likelihood,self.time_likelohood,self.type_likelihood,
                       self.cross_entropy_loss,self.SE_loss,
                       self.l2_norm,self.merged,self.train_op]

        outputs = sess.run(output_feed, input_dic)

        return outputs

    def metrics_likelihood(self,sess,batch_data):

        output_feed = [
                       self.predict_type_prob,self.labels,
                       self.log_likelihood,
                       self.time_likelohood,self.type_likelihood,self.cross_entropy_loss,self.SE_loss]
        input_dic = self.embedding.make_feed_dic(batch_data = batch_data)
        input_dic[self.now_batch_size] = len(batch_data)
        outputs = sess.run(output_feed,input_dic)
        return outputs

    def output(self):

        with tf.name_scope('loss_function'):

            self.l2_norm = tf.add_n([
                tf.nn.l2_loss(self.type_lst_embedding)
            ])
            regulation_rate = self.FLAGS.regulation_rate
            one_hot_type = tf.one_hot(
                self.target_type, depth = self.FLAGS.type_num, dtype = tf.float32
            )
            one_hot_type = tf.reshape(one_hot_type,[-1,self.FLAGS.type_num ]) # batch_size, type_num



            """type"""
            # self.predict_type_prob, batch_size,  num_units
            self.predict_type_prob = tf.matmul(self.predict_type_prob,self.embedding.type_emb_lookup_table[:-3,:],transpose_b=True)
            self.predict_type_prob = tf.nn.softmax(self.predict_type_prob)
            log_probs = tf.log (self.predict_type_prob + 1e-9)
            self.cross_entropy_loss = -tf.reduce_sum(log_probs * one_hot_type, axis=[-1])  # batch_size,

            """time"""
            self.SE_loss  = (self.target_time-tf.squeeze(self.predict_time)) ** 2  # batch_size,

            """lambda"""
            # target lambda
            # self.target_lambda = tf.matmul(self.target_lambda,self.embedding.type_emb_lookup_table[:-3,:],transpose_b=True)
            # self.target_lambda = tf.nn.relu6(self.target_lambda)

            target_type_lambda = self.target_lambda * one_hot_type  # batch_size, type_num
            log_target_type_lambda = tf.log(tf.reduce_sum(target_type_lambda+1e-9, axis=1))  # batch_size,
            sum_lambda = tf.reduce_sum(self.target_lambda ,axis = 1) # batch_size,
            log_sum_lambda = tf.log(sum_lambda)
            # self.log_target_type_lambda = log_target_type_lambda

            # sims lambda
            sum_sims_lambda = tf.reduce_sum(tf.reduce_sum(self.sims_lambda, axis=2),axis=1) # batch_size,

            integral_sims_lambda = (sum_sims_lambda/tf.to_float(self.FLAGS.sims_len)) * self.T_lst

            self.log_likelihood =  (log_target_type_lambda - integral_sims_lambda) # batch_size
            self.type_likelihood = (log_target_type_lambda - log_sum_lambda) # batch_size,
            self.time_likelohood = self.log_likelihood - self.type_likelihood

            self.log_likelihood_loss = -self.log_likelihood

            """losss"""
            if self.FLAGS.loss == 'cross_entropy':
                self.loss = tf.reduce_mean(self.cross_entropy_loss)
            elif self.FLAGS.loss == 'log_likelihood':
                self.loss =    tf.reduce_mean(self.log_likelihood_loss)
            elif self.FLAGS.loss == 'llh_ce':
                self.loss =    tf.reduce_mean(self.log_likelihood_loss)  \
                              + tf.reduce_mean(self.cross_entropy_loss)
            else: # TODO se 是否需要除以一个scale
                scale = 1
                self.loss = tf.reduce_mean(self.SE_loss)/scale\
                            + self.llh_decay_rate * tf.reduce_mean(self.log_likelihood_loss) \
                            + tf.reduce_mean(self.cross_entropy_loss)
            # self.loss = tf.reduce_mean(self.log_likelihood_loss)
            # self.loss = tf.reduce_mean(self.SE_loss)
            # self.loss = tf.reduce_mean(self.SE_loss)   \
            #             + tf.reduce_mean(self.cross_entropy_loss)
            # self.loss = 10* tf.reduce_mean(self.SE_loss) \
            #             +self.llh_decay_rate *tf.reduce_mean(self.log_likelihood_loss) \
            #             + tf.reduce_mean(self.cross_entropy_loss)
            # for metrics
            self.labels = one_hot_type
            # self.target_lambda = target_lambda

            tf.summary.scalar('l2_norm', self.l2_norm)
            tf.summary.scalar('learning_rate', self.learning_rate)
            tf.summary.scalar('llh_decay_rate', self.llh_decay_rate)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('seq_log_likelihood', tf.reduce_mean(self.log_likelihood) )
            tf.summary.scalar('time_log_likelihood', tf.reduce_mean(self.time_likelohood) )
            tf.summary.scalar('type_log_likelihood', tf.reduce_mean(self.type_likelihood) )
            tf.summary.scalar('cross_entropy_loss', tf.reduce_mean(self.cross_entropy_loss) )
            tf.summary.scalar('sqrt_mean_square_error_loss', tf.sqrt(tf.reduce_mean(self.SE_loss) ))
        self.cal_gradient(tf.trainable_variables())
