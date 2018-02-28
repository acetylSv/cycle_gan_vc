'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/dc_tts
'''
import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import librosa
import h5py
from copy import deepcopy
import librosa
from hyperparams import Hyperparams as hp

def get_spectrograms(y, sr):
    # Trimming
    y, _ = librosa.effects.trim(y)

    # stft. D: (1+n_fft//2, T)
    D = librosa.stft(
            y=y,
            n_fft=hp.n_fft,
            hop_length=hp.hop_length,
            win_length=hp.win_length
        )

    # magnitude spectrogram
    magnitude = np.abs(D) #(1+n_fft/2, T)

    # power spectrogram
    power = magnitude**2 #(1+n_fft/2, T)

    # mel spectrogram
    S = librosa.feature.melspectrogram(S=power, n_mels=hp.n_mels) #(n_mels, T)

    # return shape: (T, n_mels), (T,1+n_fft/2)
    mel = np.transpose(S.astype(np.float32))
    mag = np.transpose(magnitude.astype(np.float32))

    mel = np.log(mel + 1e-12)
    mag = np.log(mag + 1e-12)

    return mel, mag

def spectrogram2wav(mag):
    mag = np.power(np.e, mag)**1.2
    wav = griffin_lim(mag)
    return wav.astype(np.float32)

def griffin_lim(spectrogram):
    '''Applies Griffin-Lim's raw.'''
    spectrogram = spectrogram.T
    X_best = deepcopy(spectrogram)
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y

def invert_spectrogram(spectrogram):
    '''
    spectrogram: [f, t]
    '''
    return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")

def plot_alignment(alignment, gs, idx):
    fig, ax = plt.subplots()
    im = ax.imshow(alignment)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im)
    plt.title('{} Steps'.format(gs))
    plt.savefig('{}/alignment_{}_{}k.png'.format(hp.log_dir, idx, gs//1000), format='png')

def my_shuffle(*args):
    randomize = np.arange(len(args[0]))
    np.random.shuffle(randomize)
    res = [x[randomize] for x in args]
    if len(res) >= 2:
        return res
    else:
        return res[0]
