import h5py
import numpy as np
from random import shuffle

from hyperparams import Hyperparams as hp

class Data_loader:
    def __init__(self, mode):
        if mode == 'train':
            self.hf = h5py.File(hp.train_hdf5_path, 'r')
        elif mode == 'test':
            self.hf = h5py.File(hp.test_hdf5_path, 'r')
        else:
            print('=====Error, Please specify mode(train, eval or test)=====')
        self.idss = list(self.hf.keys())
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
        all_mcs = []
        all_f0s = []
        all_aps = []
        all_mc_mean = []
        all_mc_std = []
        all_logf0_mean = []
        all_logf0_std = []
        for ids in idss:
            uttr_ids = [x for x in self.hf['train'][ids]]
            for uttr_id in uttr_ids:
                mcs = self.hf['train'][ids][uttr_id]['normed_mc'][:]
                f0s = self.hf['train'][ids][uttr_id]['normed_logf0'][:]
                aps = self.hf['train'][ids][uttr_id]['ap'][:]
                
                all_mcs.extend(mcs)
                all_f0s.extend(f0s)
                all_aps.extend(aps)
                all_mc_mean.extend(self.hf['train'][ids][uttr_id]['mc_mean'][:])
                all_mc_std.extend(self.hf['train'][ids][uttr_id]['mc_std'][:])
                all_logf0_mean.extend(self.hf['train'][ids][uttr_id]['logf0_mean'][:])
                all_logf0_std.extend(self.hf['train'][ids][uttr_id]['logf0_std'][:])
        return np.array(all_mcs), np.array(all_f0s), np.array(all_aps), \
                np.array(all_mc_mean), np.array(all_mc_std), \
                np.array(all_logf0_mean), np.array(all_logf0_std)

    def get_test_partition(self, ids, uttr_id):
        mcs = self.hf['test'][ids][uttr_id]['normed_mc'][:]
        f0s = self.hf['test'][ids][uttr_id]['normed_logf0'][:]
        aps = self.hf['test'][ids][uttr_id]['ap'][:]

        mc_mean = self.hf['test'][ids][uttr_id]['mc_mean'][:]
        mc_std = self.hf['test'][ids][uttr_id]['mc_std'][:]
        logf0_mean = self.hf['test'][ids][uttr_id]['logf0_mean'][:]
        logf0_std = self.hf['test'][ids][uttr_id]['logf0_std'][:]
        
        mcs = np.expand_dims(np.reshape(mcs, [-1, hp.mcep_dim]), 0)
        f0s = np.expand_dims(np.reshape(f0s, [-1]), 0)
        aps = np.expand_dims(np.reshape(aps, [-1, 1+hp.n_fft//2]), 0)

        mc_mean = np.expand_dims(np.reshape(mc_mean, [-1, hp.mcep_dim])[0], 0)
        mc_std = np.expand_dims(np.reshape(mc_std, [-1, hp.mcep_dim])[0], 0)
        logf0_mean = np.expand_dims(np.reshape(logf0_mean, [-1])[0], -1)
        logf0_std = np.expand_dims(np.reshape(logf0_std, [-1])[0], -1)
        return np.array(mcs), np.array(f0s), np.array(aps), \
                np.array(mc_mean), np.array(mc_std), \
                np.array(logf0_mean), np.array(logf0_std)

    def close_hdf5(self):
        self.hf.close()
        return

if __name__ == '__main__':
    dl = Data_loader(mode='train')
    a, b, c, d, e, f, g = dl.get_partition(['226', '227', '232', '237'])
    print(a.shape, b.shape, c.shape, d.shape, e.shape, f.shape, g.shape)
