# -*- coding: utf-8 -*-
# @Time    : 2020/8/30 13:35
# @Author  : zxl
# @FileName: AnalyzeData_json.py

import pickle

if __name__ == "__main__":

    root = "D://Project/TPP_V2/data/origin_data/data_so/fold5/"
    sample_count = 0
    train_test_count = [0,0,0]
    train_test_seq_len = [0,0,0]
    total_seq_len = 0
    max_seq_len = 0
    min_seq_len = 1000
    type_dic = {}

    for idx, filename in enumerate(['train.pkl','dev.pkl','test.pkl']):
        file_path = root + filename
        key = filename[:-4]


        with open(file_path,'rb') as f:
            json_obj = pickle.load(f,encoding='latin-1')
        seq_lst = json_obj[key]

        for seq in seq_lst:
            sample_count += 1
            train_test_seq_len[idx]+=1
            train_test_count[idx]+= len(seq)
            total_seq_len += len(seq)
            max_seq_len =  max(max_seq_len,len(seq))
            min_seq_len = min(min_seq_len,len(seq))
            for event in seq:
                type = event['type_event']
                if type not in type_dic:
                    type_dic[type] = 0
                type_dic[type] += 1


    print('seq count: %d,event_count: %d, train count : %d, dev count : %d,test count: %d,type_num:%d, max seq len: %d, min seq len: %d,avg seq len: %.2f' % \
          (sample_count, sum(train_test_count),train_test_count[0],train_test_count[1],train_test_count[2],len(type_dic), max_seq_len, min_seq_len, total_seq_len / sample_count))
    print('train_seq_len: %d, dev_seq_len: %d, test_seq_len: %d'%(train_test_seq_len[0],train_test_seq_len[1],train_test_seq_len[2]))
