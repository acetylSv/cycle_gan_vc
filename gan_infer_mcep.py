import sys, os
import tensorflow as tf
import numpy as np

from network import *
from hyperparams import Hyperparams as hp
from utils import *
from data_loader import *
from cycle_gan_graph import Graph

def test():
    # Data loader
    dl = Data_loader(mode='test')
    # Build graph
    g = Graph(mode='test'); print("Testing Graph loaded")
    # Saver
    saver = tf.train.Saver(max_to_keep = 5)
    # Session
    sess = tf.Session()
    # If model exist, restore, else init a new one
    ckpt = tf.train.get_checkpoint_state(hp.log_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("=====Reading model parameters from %s=====" % ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        gs = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
    else:
        print("=====Error: model not found=====")
        dl.close_hdf5()
        sess.close()
        return

    # ALL DATA
    #A_idss = ['226', '227', '232', '237']
    #B_idss = ['225', '228', '229', '230']
    
    GV_A_to_B = []
    GV_B_to_A = []

    # Test In-Domain
    A_idss = ['225']
    B_idss = ['228']
    A_uttrs, B_uttrs = dl.get_uttrs(A_idss[0], B_idss[0])
    max_uttr_num = max(len(A_uttrs), len(B_uttrs))
    for idx in range(max_uttr_num):
        A_uttr, B_uttr = A_uttrs[idx%len(A_uttrs)], B_uttrs[idx%len(B_uttrs)]
        A_normed_mcs, A_normed_logf0s, A_aps, \
            A_mc_mean, A_mc_std, A_logf0_mean, A_logf0_std = \
                dl.get_test_partition(A_idss[0], A_uttr)
        B_normed_mcs, B_normed_logf0s, B_aps, \
            B_mc_mean, B_mc_std, B_logf0_mean, B_logf0_std = \
                dl.get_test_partition(B_idss[0], B_uttr)
        
        #print(A_normed_mcs.shape, A_normed_logf0s.shape, A_aps.shape)
        #print(A_mc_mean.shape, A_mc_std.shape, A_logf0_mean.shape, A_logf0_std.shape)
        mcep_A_to_B, B_mc_mean, B_mc_std, mcep_B_to_A, A_mc_mean, A_mc_std = \
            sess.run(
                [g.B_y_hat, g.B_mc_mean, g.B_mc_std, g.A_y_hat, g.A_mc_mean, g.A_mc_std],
                feed_dict={
                    g.A_x:A_normed_mcs, g.B_x:B_normed_mcs,
                    g.A_f0: A_normed_logf0s, g.B_f0: B_normed_logf0s,
                    g.A_ap: A_aps, g.B_ap: B_aps,
                    g.A_mc_mean: A_mc_mean, g.A_mc_std: A_mc_std,
                    g.B_mc_mean: B_mc_mean, g.B_mc_std: B_mc_std,
                    g.A_logf0s_mean: A_logf0_mean, g.A_logf0s_std: A_logf0_std,
                    g.B_logf0s_mean: B_logf0_mean, g.B_logf0s_std: B_logf0_std
                }
            )
        print(mcep_A_to_B.shape, B_mc_mean.shape, B_mc_std.shape)
        GV_A_to_B.extend(np.squeeze((mcep_A_to_B*B_mc_std) + B_mc_mean))
        GV_B_to_A.extend(np.squeeze((mcep_B_to_A*A_mc_std) + A_mc_mean))
    print(np.var(GV_A_to_B), np.var(GV_B_to_A))
    # exit
    dl.close_hdf5()
    sess.close()

if __name__ == '__main__':
    test()
    print('Infer Done')
