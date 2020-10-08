# -*- coding: utf-8 -*-
# @Time    : 2020/9/20 20:33
# @Author  : zxl
# @FileName: ScaleTime.py

import pickle


if __name__ == "__main__":

    root = "D://Data/Hawkes/data_meme/"


    for idx, filename in enumerate(['train.pkl','dev.pkl','test.pkl']):
        file_path = root + filename
        key = filename[:-4]
        out_path = root+key+'_small_time_normalize.pkl'


        with open(file_path,'rb') as f:
            json_obj = pickle.load(f,encoding='latin-1')
        seq_lst = json_obj[key]

        new_seq_lst = []

        for seq in seq_lst:
            # # TODO 删除小于2的序列
            # if len(seq) <2:
            #     continue
            scale = seq[-1]['time_since_start']
            if scale == 0 and len(seq) >= 1:
                continue

            new_seq= []
            for event in seq:

                event['time_since_start'] = event['time_since_start']/scale
                event['time_since_last_event'] = event['time_since_last_event'] / scale
                new_seq.append(event)
            new_seq_lst.append(new_seq)
        json_obj[key] = new_seq_lst

        pickle.dump(json_obj, open(out_path, 'wb'))

