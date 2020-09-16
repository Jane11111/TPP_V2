# -*- coding: utf-8 -*-
# @Time    : 2020/7/24 19:29
# @Author  : zxl
# @FileName: data_loader.py

import pickle
import numpy as np

class DataLoader():

    def __init__(self,FLAGS):
        self.FLAGS = FLAGS
        self.FLAGS = FLAGS
        self.user_count = 1
        self.item_count = 1
        self.category_count = 5
    def pro_time_method(self, time_stamp_seq, mask_time):
        timelast_list = [time_stamp_seq[i+1]-time_stamp_seq[i] for i in range(0,len(time_stamp_seq)-1,1)]
        timelast_list.insert(0,0)
        timenow_list = [mask_time-time_stamp_seq[i] for i in range(0,len(time_stamp_seq),1)]

        return [timelast_list,timenow_list]

    def load_data(self,file_path,train):
        data_set = []
        count =0
        max_type = 0

        with open(file_path , 'r') as f:
            l = f.readline()
            while l:

                tmp = eval(l)
                # max_type = max( max_type,np.max(tmp[0]))
                # if tmp[4]  < 2:
                #     l = f.readline()
                #     continue
                # tmp.append(self.pro_time_method(tmp[1],tmp[3]))
                # sim_time_list = []
                # for t in tmp[7]: # sims_time_lst
                #     sim_time_list.append(self.pro_time_method(tmp[1],t))
                #
                # tmp.append(sim_time_list)

                data_set.append(tmp)
                l = f.readline()
                count = count+1
        print("max type: %d"%max_type)
        return data_set

    def load_train_test(self):


        origin_train_path = self.FLAGS.in_data_root_path + "train.pkl"
        origin_test_path =  self.FLAGS.in_data_root_path + "test.pkl"

        train_path = self.FLAGS.out_data_root_path + "train.txt"
        test_path = self.FLAGS.out_data_root_path + "test.txt"

        if self.FLAGS.split_data:
            self.write_file(origin_train_path, train_path, 'train')
            self.write_file(origin_test_path, test_path, 'test')

        train_set = self.load_data(train_path,True)
        test_set = self.load_data(test_path,False)
        print('train len: %d'%len(train_set))
        print('test len: %d'%len(test_set))

        return train_set, test_set

    def process_data(self,seq_lst):
        seq_id = 0
        dataset = []
        max_seq_len = 0
        for event_seq in seq_lst:
            if len(event_seq) == 0:
                continue
            max_seq_len = max(max_seq_len, len(event_seq))
            res = self.process_seq(event_seq, seq_id)
            dataset.extend(res)
            seq_id += 1

        return max_seq_len, dataset

    def write_file(self,in_file, out_file, key):
        with open(in_file, 'rb') as f:
            dic = pickle.load(f, encoding='latin-1')
        seq_lst = dic[key]
        max_seq_len, dataset = self.process_data(seq_lst)
        print(key + ' max_seq_len : %d' % max_seq_len)
        with open(out_file, 'w') as w:
            for seq in dataset:
                w.write(str(seq) + '\n')

    def process_seq(self, event_seq, seq_id):
        """
        把单个事件序列切分成MTAM需要的形式
        :param event_seq:单个事件sequence
        :param seq_id:
        :return:
        """
        res = []
        complete_time_lst = []
        complete_type_lst = []

        for item in event_seq:
            complete_time_lst.append(item['time_since_start'])
            complete_type_lst.append(item['type_event'])

        sims_len = self.FLAGS.sims_len

        #
        for i in np.arange(1,len(complete_time_lst),1):
            target_type = complete_type_lst[i]
            target_time = complete_time_lst[i]

            not_first = 1.

            history_len = min(i,self.FLAGS.max_seq_len-1)
            start_idx = max(0, i-history_len)
            end_idx = i
            type_lst = complete_type_lst[start_idx:end_idx]
            time_lst = complete_time_lst[start_idx:end_idx]

            type_lst.append(self.FLAGS.type_num) # 最后一个是mask
            time_lst.append(target_time) # 补齐
            if self.FLAGS.integral_cal == 'MC':
                sims_time_lst = list(np.random.uniform(time_lst[-2],target_time, sims_len))
            else:
                sims_time_lst = [complete_time_lst[i-1],target_time]

            target_now_last_lst = self.pro_time_method(time_lst, target_time)
            sim_now_last_lst = []
            for t in sims_time_lst:  # sims_time_lst
                sim_now_last_lst.append(self.pro_time_method(time_lst, t))


            res.append([type_lst, time_lst,
                        target_type, target_time, len(time_lst),target_time-time_lst[-2],
                        not_first,sims_time_lst,target_now_last_lst,sim_now_last_lst])

        return res




# if __name__ == "__main__":
#
#     loder = DataLoader()
#     train_set, test_set = load_train_test()