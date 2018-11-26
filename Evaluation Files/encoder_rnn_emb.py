from __future__ import absolute_import, division, unicode_literals

import sys
import logging
import numpy as np
import time
import os

PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

# SentEval prepare and batcher
def prepare(params, samples):
    return samples

def encoder(*args):
    print(args.__dict__)

def batcher(params, batch):
    pass


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 256,
                                 'tenacity': 3, 'epoch_size': 2}

params_senteval['gensen'] = encoder

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = [
                      # 'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      # 'TREC', 
                      # 'STSBenchmark'
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 
                      'MRPC',
                      'SICKEntailment',
                       'SICKRelatedness'
                      ]
                      # 'Length', 'WordContent', 'Depth', 'TopConstituents',
                      # 'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      # 'OddManOut', 'CoordinationInversion']
    results = se.eval(transfer_tasks)
    print(results)
