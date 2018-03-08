import sys, os
import tensorflow as tf
import numpy as np

from network import *
from hyperparams import Hyperparams as hp
from utils import *
from data_loader import *
from cycle_gan_graph import Graph

# init random_seed
#tf.set_random_seed(2401)
#np.random.seed(2401)
#random.seed(2401)

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
    
    # Test In-Domain
    A_idss = ['226']
    B_idss = ['225']
    A_uttrs = ['335', '336', '337', '338', '339']
    B_uttrs = ['330', '331', '332', '334', '335']
 
    for A_uttr, B_uttr in zip(A_uttrs, B_uttrs):
        A_normed_mcs, A_normed_logf0s, A_aps, \
            A_mc_mean, A_mc_std, A_logf0_mean, A_logf0_std = \
                dl.get_test_partition(A_idss[0], A_uttr)
        B_normed_mcs, B_normed_logf0s, B_aps, \
            B_mc_mean, B_mc_std, B_logf0_mean, B_logf0_std = \
                dl.get_test_partition(B_idss[0], B_uttr)
        
        #print(A_normed_mcs.shape, A_normed_logf0s.shape, A_aps.shape)
        #print(A_mc_mean.shape, A_mc_std.shape, A_logf0_mean.shape, A_logf0_std.shape)
        
        audio_A, audio_B, audio_A_to_B, audio_B_to_A, audio_A_to_B_to_A, audio_B_to_A_to_B = \
            sess.run(
                [g.audio_A, g.audio_B, \
                 g.audio_A_to_B, g.audio_B_to_A, \
                 g.audio_A_to_B_to_A, g.audio_B_to_A_to_B],
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
        librosa.output.write_wav('test_result/in_domain/test_A_{}.wav'.format(A_idss[0]+'_'+A_uttr), np.array(audio_A), hp.sr)
        librosa.output.write_wav('test_result/in_domain/test_B_{}.wav'.format(B_idss[0]+'_'+B_uttr), np.array(audio_B), hp.sr)
        librosa.output.write_wav('test_result/in_domain/test_A_to_B_{}.wav'.format(A_idss[0]+'_'+A_uttr),
            np.array(audio_A_to_B), hp.sr)
        librosa.output.write_wav('test_result/in_domain/test_B_to_A_{}.wav'.format(B_idss[0]+'_'+B_uttr),
            np.array(audio_B_to_A), hp.sr)
        librosa.output.write_wav('test_result/in_domain/test_A_to_B_to_A_{}.wav'.format(A_idss[0]+'_'+A_uttr),
            np.array(audio_A_to_B_to_A), hp.sr)
        librosa.output.write_wav('test_result/in_domain/test_B_to_A_to_B_{}.wav'.format(B_idss[0]+'_'+B_uttr),
            np.array(audio_B_to_A_to_B), hp.sr)
    
    # Test Out-of-Domain
    A_idss = ['227', '232', '237']
    B_idss = ['228', '229', '230']
    
    for A_ids, B_ids in zip(A_idss, B_idss):
        A_uttrs, B_uttrs = dl.get_uttrs(A_ids, B_ids)
        for A_uttr, B_uttr in zip(A_uttrs, B_uttrs):
            A_normed_mcs, A_normed_logf0s, A_aps, \
                A_mc_mean, A_mc_std, A_logf0_mean, A_logf0_std = \
                    dl.get_test_partition(A_ids, A_uttr)
            B_normed_mcs, B_normed_logf0s, B_aps, \
                B_mc_mean, B_mc_std, B_logf0_mean, B_logf0_std = \
                    dl.get_test_partition(B_ids, B_uttr)
            
            #print(A_normed_mcs.shape, A_normed_logf0s.shape, A_aps.shape)
            #print(A_mc_mean.shape, A_mc_std.shape, A_logf0_mean.shape, A_logf0_std.shape)
            
            audio_A, audio_B, audio_A_to_B, audio_B_to_A, audio_A_to_B_to_A, audio_B_to_A_to_B = \
                sess.run(
                    [g.audio_A, g.audio_B, \
                     g.audio_A_to_B, g.audio_B_to_A, \
                     g.audio_A_to_B_to_A, g.audio_B_to_A_to_B],
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
            librosa.output.write_wav('test_result/out_domain/test_A_{}.wav'.format(A_idss[0]+'_'+A_uttr), np.array(audio_A), hp.sr)
            librosa.output.write_wav('test_result/out_domain/test_B_{}.wav'.format(B_idss[0]+'_'+B_uttr), np.array(audio_B), hp.sr)
            librosa.output.write_wav('test_result/out_domain/test_A_to_B_{}.wav'.format(A_idss[0]+'_'+A_uttr),
                np.array(audio_A_to_B), hp.sr)
            librosa.output.write_wav('test_result/out_domain/test_B_to_A_{}.wav'.format(B_idss[0]+'_'+B_uttr),
                np.array(audio_B_to_A), hp.sr)
            librosa.output.write_wav('test_result/out_domain/test_A_to_B_to_A_{}.wav'.format(A_idss[0]+'_'+A_uttr),
                np.array(audio_A_to_B_to_A), hp.sr)
            librosa.output.write_wav('test_result/out_domain/test_B_to_A_to_B_{}.wav'.format(B_idss[0]+'_'+B_uttr),
                np.array(audio_B_to_A_to_B), hp.sr)
    
    # exit
    dl.close_hdf5()
    sess.close()

if __name__ == '__main__':
    test()
    print('Infer Done')
