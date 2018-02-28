import h5py
import numpy as np
from random import shuffle

from hyperparams import Hyperparams as hp

class Data_loader:
    def __init__(self, mode):
        if mode == 'train':
            self.hf = h5py.File(hp.train_hdf5_path, 'r')
        elif mode == 'eval':
            self.hf = h5py.File(hp.eval_hdf5_path, 'r')
        else:
            print('=====Error, Please specify mode(train, eval or test)=====')
        self.idss = list(self.hf[mode].keys())
        self.mode = mode
        self.chunk_size = len(self.idss) // hp.partition

    def shuffle_idss(self):
        shuffle(self.idss)
        return

    def get_idss_parts(self):
        idss_part = \
           [self.idss[i:i+self.chunk_size] for i in range(0, len(self.idss), self.chunk_size)]
        return idss_part

    def get_partition(self, idss):
        all_mags = []
        for ids in idss:
            uttr_ids = [x for x in self.hf[self.mode][ids]]
            for uttr_id in uttr_ids:
                one_uttr = self.hf[self.mode][ids][uttr_id]['lin'][:]
                for idx in range(one_uttr.shape[0]//hp.fix_seq_length):
                    all_mags.append(one_uttr[idx*hp.fix_seq_length:(idx+1)*hp.fix_seq_length])
        return np.array(all_mags)

    def close_hdf5(self):
        self.hf.close()
        return

if __name__ == '__main__':
    dl = Data_loader(mode='train')
    a = dl.get_partition(['226', '227', '232', '237'])
    print(a.shape)
