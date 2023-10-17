# coding=gbk
import numpy as np
import torch
import random
from config import arg_parse, Line_Graph
args = arg_parse()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    


if __name__ == '__main__':
  for swt in [1]: 
    for p in [0]:
    #for pool in ["sum"]:
      for epoch in [40]: 
        dataset = 'MUTAG'
        pool = "mean"
        log = './result/{}_'.format(dataset)+'{}.log'.format(swt)
        with open(log, 'a+') as f:
            f.write('#######DATASET:{}############\n'.format(dataset))
            f.write("########SGN is {} #############\n".format(swt))
            f.write("########epoch is {} #############\n".format(epoch))
            f.write("#######pool is {} #############\n".format(pool))
            #f.write("#######p is {} #############\n".format(p))
            f.closed
        ac = 0
        max_acc= 0
        min_acc = 1
        for seed in [0,1,2,3,4]:
            setup_seed(seed)
            test_acc, test_log = main(dataset, swt, epoch, seed,pool, p)
            ac = ac + test_acc
            if max_acc <= test_acc:
                max_acc = test_acc
            if min_acc >= test_acc:
                min_acc = test_acc
            if seed == 0:
                test_log_seed = test_log
            else:
                for i, test_log_ in enumerate(test_log):
                    test_log_seed[i] = test_log_seed[i] + test_log_
                    
            with open(log, 'a+') as f:
                f.write("########seed:{}, test acc is {}#############\n".format(seed, test_acc))
                f.closed
        bias = max(max_acc - ac/5,ac/5- min_acc)
        print("########mean test acc is {}#############\n".format(ac/5))
        with open(log, 'a+') as f:
            f.write("########mean test acc #############\n")
            f.write("%.4f"% (ac/5)+"±"+"%.4f"%bias)
            f.write("######## avg is {}#############\n".format( [i/5 for i in test_log_seed] ))
            f.closed


