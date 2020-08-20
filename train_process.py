# -*- coding: utf-8 -*-
# @Time    : 2020-08-19 15:03
# @Author  : zxl
# @FileName: train_process.py


import os
import time
import random
import numpy as np
import tensorflow as tf

from Embedding.Type_Embedding import Type_embedding
from util.model_log import create_log
from DataHandle.get_input_data import DataInput

from Prepare.data_loader import DataLoader
from config.model_parameter import model_parameter
from Model.AttentionTPP import AttentionTPP_MLT,AttentionTPP

from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, accuracy_score

random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)

class Train_main_process:

    def __init__(self):

        start_time = time.time()
        model_parameter_ins = model_parameter()
        data_name = model_parameter_ins.flags.FLAGS.data_name
        self.FLAGS = model_parameter_ins.get_parameter(data_name).FLAGS

        log_ins = create_log(data_name = data_name, model_name = self.FLAGS.model_name, lr = self.FLAGS.learning_rate)

        self.logger = log_ins.logger
        self.logger.info("hello world the experiment begin")
        self.logger.info("the model parameter is : " + str(self.FLAGS.flag_values_dict()))

        prepare_data_ins = DataLoader(self.FLAGS)

        self.logger.info("start loading dataset!")
        self.train_set, self.test_set = prepare_data_ins.load_train_test()

        self.logger.info("dataset loaded!")

        self.logger.info("DataHandle Process cost time: %.2fs" %(time.time() - start_time))
        start_time = time.time()

        self.emb = Type_embedding(is_training=self.FLAGS.is_training,
                                  type_num = self.FLAGS.type_num,
                                  max_seq_len = self.FLAGS.max_seq_len,
                                  sims_len = self.FLAGS.sims_len,
                                  FLAGS = self.FLAGS
                                  )
        self.logger.info('get train test data process cost: %.2fs'%(time.time() - start_time))



    def train(self):
        start_time = time.time()

        if self.FLAGS.per_process_gpu_memory_fraction == 0.0:
            gpu_option = tf.GPUOptions(allow_growth = True)
        elif self.FLAGS.per_process_gpu_memory_fraction == 1.0:
            gpu_option = tf.GPUOptions()
        else:
            gpu_option = tf.GPUOptions(
                per_process_gpu_memory_fraction = self.FLAGS.per_process_gpu_memory_fraction, allow_growth = True
            )

        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_DEVISIBLE_DEVICES'] = self.FLAGS.cuda_visible_devices


        self.sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_option))

        if not tf.test.gpu_device_name():
            self.logger.warning('NO GPU is FOUND')
        else:
            self.logger.info(tf.test.gpu_device_name())


        with self.sess.as_default():

            if self.FLAGS.model_name == 'AttentionTPP_MLT':
                self.model = AttentionTPP_MLT(self.FLAGS, self.emb, self.sess)
            elif self.FLAGS.model_name == 'AttentionTPP':
                self.model = AttentionTPP(self.FLAGS, self.emb, self.sess)
            self.logger.info('Init finish. cost time: %.2fs' %(time.time() - start_time))



            def eval_model():

                total_event_num = len(self.test_set)
                type_prob =  []
                target_type = [] # one_hot 形式的
                for step_i, batch_data in DataInput(self.test_set,self.FLAGS.test_batch_size):
                    step_type_prob, step_target_type = self.model.metrics_likelihood(sess = self.sess,
                                                                       batch_data = batch_data)
                    type_prob.extend(step_type_prob)
                    target_type.extend(step_target_type)

                predict_type = []
                for prob_arr in type_prob:
                    idx = np.argmax(prob_arr)
                    arr = np.zeros(shape=(self.FLAGS.type_num,))
                    arr[idx] = 1
                    predict_type.append(arr)
                target_type = np.array(target_type)
                predict_type = np.array(predict_type)
                auc = roc_auc_score(target_type, predict_type,average='micro')
                f1 = f1_score(target_type, predict_type,average='micro')
                recall = recall_score(target_type,predict_type,average='micro')
                precision = precision_score(target_type,predict_type,average='micro')
                accuracy = accuracy_score(target_type, predict_type, )

                return auc, f1, recall, precision, accuracy


            self.logger.info('learning rate: %f'%(self.FLAGS.learning_rate))
            self.logger.info('train set: %d'%len(self.train_set))

            self.global_step = 0
            avg_loss = 0.0
            learning_rate = self.FLAGS.learning_rate

            for epoch in range(self.FLAGS.max_epochs):

                random.shuffle(self.train_set)
                for step_i,train_batch_data in DataInput(self.train_set, self.FLAGS.train_batch_size):

                    self.global_step += 1

                    step_loss, merge = self.model.train(self.sess, train_batch_data, learning_rate)
                    self.model.train_writer.add_summary(merge, self.global_step)

                    avg_loss += step_loss


                    if self.global_step % self.FLAGS.display_freq == 0:
                        self.logger.info("epoch: %d, step: %d, global_step: %d, train_loss :%.2f"
                                         %(epoch, step_i, self.global_step,avg_loss/ self.FLAGS.display_freq))
                        avg_loss = 0.0

                    if self.global_step % self.FLAGS.eval_freq == 0:
                        auc, f1, recall, precision, accuracy = eval_model()
                        self.logger.info("auc: %.2f" % (auc))
                        self.logger.info("f1: %.2f" % (f1))
                        self.logger.info("recall: %.2f" % (recall))
                        self.logger.info("precision: %.2f" % (precision))
                        self.logger.info("accuracy: %.2f" % (accuracy))

                self.logger.info("epoch : %d"%(epoch))
                auc, f1, recall, precision, accuracy = eval_model()
                self.logger.info("auc: %.2f" % (auc))
                self.logger.info("f1: %.2f" % (f1))
                self.logger.info("recall: %.2f" % (recall))
                self.logger.info("precision: %.2f" % (precision))
                self.logger.info("accuracy: %.2f" % (accuracy))


            self.save_model()




    def save_model(self):
         is_save_model = False

         if self.global_step % 2000 == 0:
             is_save_model = True

         if is_save_model:
             path = "D://Project/TPP/check_point/" + self.FLAGS.data_name + '-' +\
                    str(self.FLAGS.learning_rate) + "/"
             self.model.save(self.sess,self.global_step, path = path)


if __name__ == "__main__":
    main_process = Train_main_process()
    main_process.train()
