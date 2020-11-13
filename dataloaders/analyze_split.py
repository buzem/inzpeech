import os

txt_dir = '/media/data/bbekci/voxceleb/iden_split.txt'

tr_idens = {}
val_idens = {}
te_idens = {}

with open(txt_dir, 'r') as identxt:
    lines = identxt.readlines()

train_paths = []
test_paths = []
val_paths = []

for line in lines:
    subset, path = line.strip().split(' ')
    if subset == '1':
        train_paths.append(path)
    elif subset == '2':
        val_paths.append(path)
    elif subset == '3':
        test_paths.append(path)

print(test_paths[:20])

for p in train_paths:
    iden, vid, aud = p.split('/')
    if iden not in tr_idens:
        tr_idens[iden] = []
    if vid not in tr_idens[iden]:
        tr_idens[iden].append(vid)

for p in test_paths:
    iden, vid, aud = p.split('/')
    if iden not in te_idens:
        te_idens[iden] = []
    if vid not in te_idens[iden]:
        te_idens[iden].append(vid)

for p in val_paths:
    iden, vid, aud = p.split('/')
    if iden not in val_idens:
        val_idens[iden] = []
    if vid not in val_idens[iden]:
        val_idens[iden].append(vid)

for k in te_idens:
    print("ID: ", k , " aud: ", len(te_idens[k]))