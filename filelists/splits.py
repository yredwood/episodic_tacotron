import os

fname = 'libritts_train_clean_100_audiopath_text_sid_shorterthan10s_atleast5min_train_filelist.txt'
# generate train-test splits

validation_sids = ['4018', '3486', '6836', '1970',
        '8123', '8051', '5808', '8108', '2952', '1069', '2416', '3526']

with open(fname, 'r') as f:
    lines = f.readlines()


train_list = []
test_list = []

for line in lines:
    sid = line.split('|')[-1].strip()
    if sid in validation_sids:
        test_list.append(line)
    else: 
        train_list.append(line)

with open('libri100_train.txt', 'w') as f:
    f.writelines(train_list)

with open('libri100_val.txt', 'w') as f:
    f.writelines(test_list)
