import os, sys
from data_loader import *
from utils import *
from hyperparams import Hyperparams as hp
import h5py
fff = h5py.File(sys.argv[1], 'r')
a = fff['train']['227']['123']['lin'][:]

#fname = sys.argv[1]
#y, sr = librosa.load(fname, sr=hp.sr, dtype=np.float64)
#mel, mag = get_spectrograms(y, sr)
wav = spectrogram2wav(a)
librosa.output.write_wav('test123.wav', wav, hp.sr)
exit()
