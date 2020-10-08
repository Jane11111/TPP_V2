# -*- coding: utf-8 -*-
# @Time    : 2020/9/21 9:54
# @Author  : zxl
# @FileName: FilterSeq.py

import pickle

"""
删除meme数据集里面长度小于5的
"""

if __name__ == "__main__":

    root = "D://Data/Hawkes/data_meme/"


    for idx, filename in enumerate(['train.pkl','dev.pkl','test.pkl']):
        file_path = root + filename
        key = filename[:-4]
        out_path = root+key+'_long_seq.pkl'


        with open(file_path,'rb') as f:
            json_obj = pickle.load(f,encoding='latin-1')
        seq_lst = json_obj[key]

        new_seq_lst = []
        for seq in seq_lst:
            # # TODO 删除小于2的序列
            if len(seq) <10:
                continue

            new_seq_lst.append(seq)
        json_obj[key] = new_seq_lst

        pickle.dump(json_obj, open(out_path, 'wb'))


