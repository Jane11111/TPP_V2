# -*- coding: utf-8 -*-
# @Time    : 2020-08-19 15:03
# @Author  : zxl
# @FileName: train_process.py


import os
import time
import random
import numpy as np
import tensorflow as tf

from Embedding.history_embedding import history_embedding
from util.model_log import create_log
from DataHandle.get_input_data import DataInput

from Prepare.data_loader import DataLoader
from config.model_parameter import model_parameter
from Model.AttentionTPP import AttentionTPP_MLT, AttentionTPP, MTAM_TPP_W, MTAM_TPP_E, \
    MTAM_TPP_wendy,MTAM_only_time_aware_RNN,Vallina_Gru
from Model.THP import THP
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
        print('test event len: %d'%(len(self.test_set)))

        self.logger.info("dataset loaded!")

        self.logger.info("DataHandle Process cost time: %.2fs" %(time.time() - start_time))
        start_time = time.time()

        self.emb = history_embedding(is_training=self.FLAGS.is_training,
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

        global_step_lr = tf.Variable(0, trainable=False)
        decay_rate = tf.train.exponential_decay(
            learning_rate=1., global_step=global_step_lr, decay_steps=100, decay_rate=0.99,
            staircase=True)

        with self.sess.as_default():

            if self.FLAGS.model_name == 'AttentionTPP_MLT':
                self.model = AttentionTPP_MLT(self.FLAGS, self.emb, self.sess)
            elif self.FLAGS.model_name == 'AttentionTPP':
                self.model = AttentionTPP(self.FLAGS, self.emb, self.sess)
            elif self.FLAGS.model_name == 'MTAM_TPP_W':
                self.model = MTAM_TPP_W(self.FLAGS, self.emb, self.sess)
            elif self.FLAGS.model_name == 'MTAM_TPP_E':
                self.model = MTAM_TPP_E(self.FLAGS, self.emb, self.sess)
            elif self.FLAGS.model_name == 'MTAM_TPP_wendy':
                self.model = MTAM_TPP_wendy(self.FLAGS, self.emb, self.sess)
            elif self.FLAGS.model_name == 'MTAM_only_time_aware_RNN':
                self.model = MTAM_only_time_aware_RNN(self.FLAGS, self.emb, self.sess)
            elif self.FLAGS.model_name == 'Vallina_Gru':
                self.model = Vallina_Gru(self.FLAGS, self.emb, self.sess)
            elif self.FLAGS.model_name == 'THP':
                self.model = THP(self.FLAGS, self.emb, self.sess)
            self.logger.info('Init finish. cost time: %.2fs' %(time.time() - start_time))



            def eval_model():

                type_prob =  []
                target_type = [] # one_hot 形式的
                seq_llh = []
                time_llh = []
                type_llh = []
                squared_error = []
                total_pred_num = len(self.test_set)
                for step_i, batch_data in DataInput(self.test_set,self.FLAGS.test_batch_size):
                    step_type_prob, step_target_type,\
                    step_seq_llh,step_time_llh,step_type_llh,\
                    step_cross_entropy, step_se_loss = self.model.metrics_likelihood(sess = self.sess,
                                                                       batch_data = batch_data)
                    type_prob.extend(list(step_type_prob))
                    target_type.extend(list(step_target_type))
                    seq_llh.extend(list(step_seq_llh))
                    time_llh.extend(list(step_time_llh))
                    type_llh.extend(list(step_type_llh))
                    squared_error.extend(list(step_se_loss))

                correct_num = 0
                for i in range(len(type_prob)):
                    pred_probs = type_prob[i]
                    truth_probs = target_type[i]
                    idx_pred = np.argmax(pred_probs)
                    idx_truth = np.argmax(truth_probs)
                    if idx_pred == idx_truth :
                        correct_num += 1

                accuracy = correct_num/total_pred_num #TODO 计算方法需更正

                avg_log_likelihood = np.mean(seq_llh)
                avg_time_llh = np.mean(time_llh)
                avg_type_llh = np.mean(type_llh)
                rmse = np.sqrt(np.mean(squared_error) )

                return avg_log_likelihood, accuracy, rmse


            self.logger.info('learning rate: %f'%(self.FLAGS.learning_rate))
            self.logger.info('train set: %d'%len(self.train_set))

            self.global_step = 0
            avg_loss = 0.0
            sum_seq_llh = 0.0
            sum_time_llh = 0.0
            sum_type_llh = 0.0
            sum_ce_loss = 0.0
            sum_se_loss = 0.0
            count = 0
            learning_rate = self.FLAGS.learning_rate

            llh_lst =[-100]
            acc_lst = [0]
            rmse_lst = [100]

            early_stop = 0


            for epoch in range(self.FLAGS.max_epochs):
                epoch_start_time = time.time()

                random.shuffle(self.train_set)
                # 内存增加原因？？？
                self.sess.graph.finalize()

                for step_i,train_batch_data in DataInput(self.train_set, self.FLAGS.train_batch_size):
                    llh_decay_rate = self.sess.run(decay_rate, feed_dict={global_step_lr: self.global_step})

                    self.global_step += 1

                    # target_time,predict_target_emb,last_time,target_lambda,\
                    # test_output,\
                    step_loss, \
                    seq_llh,time_llh,type_llh,\
                    ce_loss,se_loss,\
                    l2_norm, merge,_ = self.model.train(self.sess, train_batch_data, learning_rate,llh_decay_rate)
                    self.model.train_writer.add_summary(merge, self.global_step)

                    # print(test_output[0])
                    count += len(train_batch_data)

                    avg_loss += step_loss
                    sum_seq_llh+=np.sum(seq_llh)
                    sum_time_llh += np.sum(time_llh)
                    sum_type_llh += np.sum(type_llh)
                    sum_ce_loss+=np.sum(ce_loss)
                    sum_se_loss += np.sum(se_loss)
                    #print("step_loss: %.5f, l2_norm: %.5f"%(step_loss,l2_norm))
                    # print(target_lambda)

                    # if self.global_step % self.FLAGS.display_freq == 0:
                    #     self.logger.info("epoch: %d, train_loss :%.5f, seq llh :%.5f,"
                    #                      "cross_entropy_loss: %.5f, rmse_loss: %.5f,"
                    #                      "time_llh: %.5f, type_llh:%.5f, step: %d, global_step: %d"
                    #                      %(epoch,avg_loss/ self.FLAGS.display_freq,
                    #                        sum_seq_llh/count,
                    #                        sum_ce_loss/count,np.sqrt(sum_se_loss/count),
                    #                        sum_time_llh/count,sum_type_llh/count,
                    #                        step_i, self.global_step))
                    #     avg_loss = 0.0
                    #     sum_seq_llh = 0.0
                    #     sum_time_llh = 0.0
                    #     sum_type_llh = 0.0
                    #     sum_ce_loss = 0.0
                    #     sum_se_loss = 0.0
                    #     count = 0.0

                self.logger.info("epoch : %d"%(epoch))
                avg_llh, accuracy,rmse  = eval_model()
                self.logger.info("log likelihood: %.5f, accuracy: %.5f, sqrt mean squared error: %.5f"
                                 % (avg_llh,accuracy,rmse))
                self.logger.info('one epoch Cost time: %.2f' % (time.time() - epoch_start_time))

                if avg_llh<=np.max(llh_lst) and accuracy <= np.max(acc_lst) and rmse>=np.min(rmse_lst):
                    early_stop += 1
                    print('llh: %.5f, accuracy: %.5f, rmse: %.5f'%(avg_llh,accuracy,rmse))
                    print('max llh: %.5f, accuracy: %.5f, rmse: %.5f' %(np.max(llh_lst),np.max(acc_lst),np.min(rmse_lst)))
                else:
                    early_stop = 0

                llh_lst.append(avg_llh)
                acc_lst.append(accuracy)
                rmse_lst.append(rmse)
                self.logger.info("MAX log likelihood: %.5f, MAX accuracy: %.5f,MIN sqrt mean squared error: %.5f"
                                 % (np.max(llh_lst), np.max(acc_lst), np.min(rmse_lst)))
                if early_stop >= 5: # 连续5轮都没有最好的结果好
                    break

                # self.save_model()

    def save_model(self):

         timeArray = time.localtime(time.time())
         timeStr = time.strftime("%Y_%m_%d__%H_%M_%S", timeArray)

         path = "D://Project/TPP_V2/check_point/" +self.FLAGS.model_name+"-"+timeStr+"-"+ self.FLAGS.data_name + '-' +\
                str(self.FLAGS.learning_rate) + "/"
         self.model.save(self.sess,self.global_step, path = path)


if __name__ == "__main__":
    main_process = Train_main_process()
    main_process.train()
