import sys, os
import tensorflow as tf
import numpy as np
import random
from random import shuffle

from network import *
from hyperparams import Hyperparams as hp
from utils import *
from data_loader import *

# Define Graph
class Graph:
    def __init__(self, mode="train"):
        self.feed_previous = True
        # Set phase
        is_training=True if mode=="train" else False

        # Set GAN Loss Criterion (Defined in module.py)
        self.criterion = mae_criterion
    

        # Input, Output Placeholder
        if mode=='test':
            # normalization term
            self.A_mc_mean = tf.placeholder(tf.float32, shape=(None, hp.mcep_dim))
            self.B_mc_mean = tf.placeholder(tf.float32, shape=(None, hp.mcep_dim))
            self.A_mc_std = tf.placeholder(tf.float32, shape=(None, hp.mcep_dim))
            self.B_mc_std = tf.placeholder(tf.float32, shape=(None, hp.mcep_dim))
            self.A_logf0s_mean = tf.placeholder(tf.float32, shape=(None))
            self.A_logf0s_std = tf.placeholder(tf.float32, shape=(None))
            self.B_logf0s_mean = tf.placeholder(tf.float32, shape=(None))
            self.B_logf0s_std = tf.placeholder(tf.float32, shape=(None))
            # input
            self.A_x = tf.placeholder(tf.float32, shape=(1, None, hp.mcep_dim))
            self.A_f0 = tf.placeholder(tf.float32, shape=(1, None))
            self.A_ap = tf.placeholder(tf.float32, shape=(1, None, 1+hp.n_fft//2))
            self.B_x = tf.placeholder(tf.float32, shape=(1, None, hp.mcep_dim))
            self.B_f0 = tf.placeholder(tf.float32, shape=(1, None))
            self.B_ap = tf.placeholder(tf.float32, shape=(1, None, 1+hp.n_fft//2))
            
            with tf.variable_scope('gen_A_to_B'):
                self.B_y_hat = build_generator(self.A_x)
            with tf.variable_scope('gen_B_to_A'):
                self.A_y_hat = build_generator(self.B_x)
            self.audio_A_to_B = tf.py_func(MCEPs2wav, [self.B_y_hat[0], self.A_f0[0], self.A_ap[0], \
                    self.B_mc_mean, self.B_mc_std, self.B_logf0s_mean, self.B_logf0s_std], tf.float32)
            self.audio_B_to_A = tf.py_func(MCEPs2wav, [self.A_y_hat[0], self.B_f0[0], self.B_ap[0], \
                    self.A_mc_mean, self.A_mc_std, self.A_logf0s_mean, self.A_logf0s_std], tf.float32)

        else:
            # normalization term
            self.A_mc_mean = tf.placeholder(tf.float32, shape=(hp.batch_size, hp.mcep_dim))
            self.B_mc_mean = tf.placeholder(tf.float32, shape=(hp.batch_size, hp.mcep_dim))
            self.A_mc_std = tf.placeholder(tf.float32, shape=(hp.batch_size, hp.mcep_dim))
            self.B_mc_std = tf.placeholder(tf.float32, shape=(hp.batch_size, hp.mcep_dim))
            self.A_logf0s_mean = tf.placeholder(tf.float32, shape=(hp.batch_size,1))
            self.A_logf0s_std = tf.placeholder(tf.float32, shape=(hp.batch_size,1))
            self.B_logf0s_mean = tf.placeholder(tf.float32, shape=(hp.batch_size,1))
            self.B_logf0s_std = tf.placeholder(tf.float32, shape=(hp.batch_size,1))
            # input
            self.A_x = tf.placeholder(tf.float32, shape=(hp.batch_size, hp.fix_seq_length, hp.mcep_dim))
            self.A_f0 = tf.placeholder(tf.float32, shape=(hp.batch_size, hp.fix_seq_length))
            self.A_ap = tf.placeholder(tf.float32, shape=(hp.batch_size, hp.fix_seq_length, 1+hp.n_fft//2))
            self.B_x = tf.placeholder(tf.float32, shape=(hp.batch_size, hp.fix_seq_length, hp.mcep_dim))
            self.B_f0 = tf.placeholder(tf.float32, shape=(hp.batch_size, hp.fix_seq_length))
            self.B_ap = tf.placeholder(tf.float32, shape=(hp.batch_size, hp.fix_seq_length, 1+hp.n_fft//2))

            # Domain-Transfering
            with tf.variable_scope('gen_A_to_B'):
                self.B_y_hat = build_generator(self.A_x)

            with tf.variable_scope('gen_B_to_A'):
                self.A_y_hat = build_generator(self.B_x)

            # Cycle-Consistency
            with tf.variable_scope('gen_A_to_B', reuse=True):
                self.B_cycle_y_hat = build_generator(self.A_y_hat)
            with tf.variable_scope('gen_B_to_A', reuse=True):
                self.A_cycle_y_hat = build_generator(self.B_y_hat)

            # Identity-Mapping
            with tf.variable_scope('gen_A_to_B', reuse=True):
                self.B_identity_y_hat = build_generator(self.B_x)
            with tf.variable_scope('gen_B_to_A', reuse=True):
                self.A_identity_y_hat = build_generator(self.A_x)

            # Discriminator
            with tf.variable_scope('dis_A') as scope:
                self.v_A_real_logits, self.v_A_real = build_discriminator(self.A_x)
                scope.reuse_variables()
                self.v_A_fake_logits, self.v_A_fake = build_discriminator(self.A_y_hat)

            with tf.variable_scope('dis_B') as scope:
                self.v_B_real_logits, self.v_B_real = build_discriminator(self.B_x)
                scope.reuse_variables()
                self.v_B_fake_logits, self.v_B_fake  = build_discriminator(self.B_y_hat)

            self.gen_vars = [v for v in tf.trainable_variables() if v.name.startswith('gen_')]
            self.dis_vars = [v for v in tf.trainable_variables() if v.name.startswith('dis_')]
            '''
            for v in self.dis_vars : print(v)
            print('----------------------')
            for v in self.gen_vars : print(v)
            '''
            '''
            vs = [v for v in tf.trainable_variables()]
            for v in vs : print(v)
            '''

            # monitor
            self.audio_A = tf.py_func(MCEPs2wav, [self.A_x[0], self.A_f0[0], self.A_ap[0], \
                    self.A_mc_mean, self.A_mc_std, self.A_logf0s_mean, self.A_logf0s_std], tf.float32)
            self.audio_B = tf.py_func(MCEPs2wav, [self.B_x[0], self.B_f0[0], self.B_ap[0], \
                    self.B_mc_mean, self.B_mc_std, self.B_logf0s_mean, self.B_logf0s_std], tf.float32)
            self.audio_A_to_B = tf.py_func(MCEPs2wav, [self.B_y_hat[0], self.A_f0[0], self.A_ap[0], \
                    self.B_mc_mean, self.B_mc_std, self.B_logf0s_mean, self.B_logf0s_std], tf.float32)
            self.audio_B_to_A = tf.py_func(MCEPs2wav, [self.A_y_hat[0], self.B_f0[0], self.B_ap[0], \
                    self.A_mc_mean, self.A_mc_std, self.A_logf0s_mean, self.A_logf0s_std], tf.float32)

            self.audio_A_to_B_to_A = tf.py_func(MCEPs2wav, \
                    [self.A_cycle_y_hat[0], self.A_f0[0], self.A_ap[0], \
                    self.A_mc_mean, self.A_mc_std, self.A_logf0s_mean, self.A_logf0s_std], tf.float32)
            self.audio_B_to_A_to_B = tf.py_func(MCEPs2wav, \
                    [self.B_cycle_y_hat[0], self.B_f0[0], self.B_ap[0], \
                    self.B_mc_mean, self.B_mc_std, self.B_logf0s_mean, self.B_logf0s_std], tf.float32)
            self.audio_A_to_A = tf.py_func(MCEPs2wav, \
                    [self.A_identity_y_hat[0], self.A_f0[0], self.A_ap[0], \
                    self.A_mc_mean, self.A_mc_std, self.A_logf0s_mean, self.A_logf0s_std], tf.float32)
            self.audio_B_to_B = tf.py_func(MCEPs2wav, \
                    [self.B_identity_y_hat[0], self.B_f0[0], self.B_ap[0], \
                    self.B_mc_mean, self.B_mc_std, self.B_logf0s_mean, self.B_logf0s_std], tf.float32)
            if mode in ("train", "eval"):
                # Loss
                ## Generator Loss
                self.loss_gen_A = self.criterion(self.v_B_fake_logits, tf.ones_like(self.v_B_fake_logits))
                self.loss_gen_B = self.criterion(self.v_A_fake_logits, tf.ones_like(self.v_A_fake_logits))
                self.loss_gen = self.loss_gen_A + self.loss_gen_B
                ## Discriminator Loss
                self.loss_dis_A = (
                    self.criterion(self.v_A_real_logits, tf.ones_like(self.v_A_real_logits)) \
                    + self.criterion(self.v_A_fake_logits, tf.zeros_like(self.v_A_fake_logits)) \
                                 ) /2
                self.loss_dis_B = (
                    self.criterion(self.v_B_real_logits, tf.ones_like(self.v_B_real_logits)) \
                    + self.criterion(self.v_B_fake_logits, tf.zeros_like(self.v_B_fake_logits)) \
                                 ) /2
                self.loss_dis = self.loss_dis_A + self.loss_dis_B
                ### Cycle Loss
                loss_cycle_A = tf.reduce_mean(tf.abs(self.A_x - self.A_cycle_y_hat))
                loss_cycle_B = tf.reduce_mean(tf.abs(self.B_x - self.B_cycle_y_hat))
                self.loss_cycle = loss_cycle_A + loss_cycle_B
                ### Identity Loss
                loss_self_identity_A = tf.reduce_mean(tf.abs(self.A_x - self.A_identity_y_hat))
                loss_self_identity_B = tf.reduce_mean(tf.abs(self.B_x - self.B_identity_y_hat))
                self.loss_identity = loss_self_identity_A + loss_self_identity_B

                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.gen_optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr*2)
                self.dis_optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr)

                with tf.variable_scope('gen_train'):
                    gvs = self.gen_optimizer.compute_gradients(
                            (self.loss_gen + \
                            hp.LAMBDA_CYCLE*self.loss_cycle + \
                            hp.LAMBDA_IDENTITY*self.loss_identity),
                            var_list=self.gen_vars
                          )
                    clipped = []
                    for grad, var in gvs:
                        grad = tf.clip_by_norm(grad, 5.)
                        clipped.append((grad, var))
                    self.train_gen_op = self.gen_optimizer.apply_gradients(
                                            clipped,
                                            global_step=self.global_step
                                        )

                with tf.variable_scope('dis_train'):
                    gvs = self.dis_optimizer.compute_gradients(
                            self.loss_dis,
                            var_list=self.dis_vars
                        )
                    clipped = []
                    for grad, var in gvs:
                        grad = tf.clip_by_norm(grad, 5.)
                        clipped.append((grad, var))
                    self.train_dis_op = self.dis_optimizer.apply_gradients(
                                            clipped,
                                            #global_step=self.global_step
                                        )
                # Summary
                tf.summary.scalar('{}/loss_dis'.format(mode), self.loss_dis)
                tf.summary.scalar('{}/loss_gen'.format(mode), self.loss_gen)
                tf.summary.scalar('{}/loss_cycle'.format(mode), self.loss_cycle)
                tf.summary.scalar('{}/loss_identity'.format(mode), self.loss_identity)
                
                tf.summary.image("{}/real_A_mag".format(mode),
                           tf.expand_dims(self.A_x, -1), max_outputs=1)
                tf.summary.image("{}/fake_A_mag".format(mode),
                           tf.expand_dims(self.A_y_hat, -1), max_outputs=1)
                tf.summary.image("{}/cycle_A_mag".format(mode),
                           tf.expand_dims(self.A_cycle_y_hat, -1), max_outputs=1)
                tf.summary.image("{}/identity_A_mag".format(mode),
                           tf.expand_dims(self.A_identity_y_hat, -1), max_outputs=1)
                tf.summary.image("{}/real_B_mag".format(mode),
                           tf.expand_dims(self.B_x, -1), max_outputs=1)
                tf.summary.image("{}/fake_B_mag".format(mode),
                           tf.expand_dims(self.B_y_hat, -1), max_outputs=1)
                tf.summary.image("{}/cycle_B_mag".format(mode),
                           tf.expand_dims(self.B_cycle_y_hat, -1), max_outputs=1)
                tf.summary.image("{}/identity_B_mag".format(mode),
                           tf.expand_dims(self.B_identity_y_hat, -1), max_outputs=1)

                tf.summary.audio("{}/A".format(mode),
                           tf.expand_dims(self.audio_A, 0), hp.sr)
                tf.summary.audio("{}/B".format(mode),
                           tf.expand_dims(self.audio_B, 0), hp.sr)
                tf.summary.audio("{}/A_to_B".format(mode),
                           tf.expand_dims(self.audio_A_to_B, 0), hp.sr)
                tf.summary.audio("{}/B_to_A".format(mode),
                           tf.expand_dims(self.audio_B_to_A, 0), hp.sr)
                tf.summary.audio("{}/A_to_B_to_A".format(mode),
                           tf.expand_dims(self.audio_A_to_B_to_A, 0), hp.sr)
                tf.summary.audio("{}/B_to_A_to_B".format(mode),
                           tf.expand_dims(self.audio_B_to_A_to_B, 0), hp.sr)
                tf.summary.audio("{}/A_to_A".format(mode),
                           tf.expand_dims(self.audio_A_to_A, 0), hp.sr)
                tf.summary.audio("{}/B_to_B".format(mode),
                           tf.expand_dims(self.audio_B_to_B, 0), hp.sr)

                self.summary_op = tf.summary.merge_all()

                # init
                self.init_op = tf.global_variables_initializer()

if __name__ == '__main__':
    g = Graph(); print('Graph Test OK')
