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

def get_batch(A_mags, B_mags, idx):
    A_batch_mags = A_mags[idx*hp.batch_size:(idx+1)*hp.batch_size]
    B_batch_mags = B_mags[idx*hp.batch_size:(idx+1)*hp.batch_size]
    return A_batch_mags, B_batch_mags

def train():
    # Data loader
    dl = Data_loader(mode='train')
    # Build graph
    g = Graph(mode='train'); print("Training Graph loaded")
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
        print("=====Init a new model=====")
        sess.run([g.init_op])
        gs = 0
    # Start training
    summary_writer = tf.summary.FileWriter(hp.log_dir, sess.graph)

    A_idss = ['226', '227', '232', '237']
    B_idss = ['225', '228', '229', '230']

    A_mags = dl.get_partition(A_idss)
    B_mags = dl.get_partition(B_idss)
    print(A_mags.shape, B_mags.shape)

    while True:
        # word-level shuffle
        A_mags = my_shuffle(A_mags)
        B_mags = my_shuffle(B_mags)
        min_seg_num = min(A_mags.shape[0], B_mags.shape[0])
        for idx in range(min_seg_num//hp.batch_size):
            # get batch
            A_batch_mags, B_batch_mags, = get_batch(A_mags, B_mags, idx)
            # Train G
            _ = sess.run(g.train_dis_op,
                    feed_dict={
                        g.A_x:A_batch_mags, g.B_x:B_batch_mags
                    }
                )
            # Train D
            loss_d_eval, loss_g_eval, loss_cycle_eval, loss_identity_eval, _, gs = \
                sess.run(
                    [g.loss_dis, g.loss_gen, g.loss_cycle, g.loss_identity, \
                                                    g.train_gen_op, g.global_step],
                    feed_dict={
                        g.A_x:A_batch_mags, g.B_x:B_batch_mags
                    }
                )

            print('%7d: Loss G : %1.6f, Loss D : %1.6f, Loss Cycle : %1.6f, Loss Identity : %1.6f' \
                % (gs, loss_g_eval, loss_d_eval, loss_cycle_eval, loss_identity_eval)
            )

            if(gs % hp.summary_period == 0):
                summary_str = sess.run(
                    g.summary_op,
                    feed_dict={
                        g.A_x:A_batch_mags, g.B_x:B_batch_mags
                    }
                )
                summary_writer.add_summary(summary_str, gs)

            if(gs % hp.save_period == 0):
                saver.save(sess, os.path.join(hp.log_dir, 'model.ckpt'), global_step=gs)
                print('Save model to %s-%d' % (os.path.join(hp.log_dir, 'model.ckpt'), gs))

    # exit
    dl.close_hdf5()
    sess.close()

if __name__ == '__main__':
    train()
    print('Training Done')
