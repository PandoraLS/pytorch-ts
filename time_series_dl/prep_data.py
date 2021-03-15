# -*- coding: utf-8 -*-
# @Time    : 2020/8/24 4:22 下午
# @Author  : sen


def AMZN_dataprep():
    """
    将csv的数据的时间戳改成序号
    Returns:
    """
    src_file_path = "/Users/seenli/Documents/workspace/code/pytorch_learn2/time_series_DL/Twitter_volume_AMZN.csv"
    tar_file_path = "/Users/seenli/Documents/workspace/code/pytorch_learn2/time_series_DL/Twitter_volume_AMZN_num.csv"

    tar_file = open(tar_file_path, 'w', encoding='utf8')
    f = open(src_file_path)
    lines = f.readlines()
    for i in range(len(lines)):
        if i == 0:
            tar_file.write(lines[i])
        else:
            new_line = lines[i].strip().split(',')[1]
            print(str(i)+','+new_line)
            tar_file.write(str(i)+','+new_line+'\n')



if __name__ == '__main__':
    AMZN_dataprep()