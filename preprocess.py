import time
import os
import sys
import pickle
from data.RNAGraph import RNAGraphDatasetDGL

start = time.time()

DATASET_NAME = sys.argv[1]
debias = sys.argv[2]

# os.chdir('../../') # go to root folder of the project
print(os.getcwd())

basedir = os.getcwd()
if debias == 'True':
    path_template = os.path.join(basedir, 'data', 'GraphProt_CLIP_sequences', 'RNAGraphProb_debias')
else:
    path_template = os.path.join(basedir, 'data', 'GraphProt_CLIP_sequences', 'RNAGraphProb')
if os.path.exists(path_template) is False:
    os.mkdir(path_template)
path_template = os.path.join(path_template, DATASET_NAME + '.pkl')
if os.path.exists(path_template) is True:
    print(DATASET_NAME + '.pkl' + " already exists!")
    exit()
with open(path_template, 'wb') as f:
    pickle.dump([], f)

dataset = RNAGraphDatasetDGL(DATASET_NAME, debias=debias)

print('Time (sec):', time.time() - start)  # 356s=6min

print(len(dataset.train))
print(len(dataset.val))
print(len(dataset.test))

print(dataset.train[0])
print(dataset.val[0])
print(dataset.test[0])

start = time.time()

with open(path_template, 'wb') as f:
    pickle.dump([dataset.train, dataset.val, dataset.test], f)

print('Time (sec):', time.time() - start)  # 38s
