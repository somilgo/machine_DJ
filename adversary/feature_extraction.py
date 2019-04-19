import librosa
import soundfile as sf
import numpy as np

STRIDE_SIZE = 10.
FRAME_SIZE = 25.
N_MFCC = 16

def get_mfcc_features(song_data, sample_rate, stride_size = STRIDE_SIZE, frame_size = FRAME_SIZE):
	mfccs = librosa.feature.mfcc(song_data, sample_rate, 
								 n_mfcc=N_MFCC,
								 hop_length=int(STRIDE_SIZE / 1000. * sample_rate), 
								 n_fft=int(FRAME_SIZE / 1000. * sample_rate))
	return mfccs

file = "../wavs/pop/pop.00008.wav"
data, samplerate = sf.read(file)
mfcc_fts = get_mfcc_features(data[:len(data)//2], samplerate)
print(len(mfcc_fts), len(mfcc_fts[0]))
mfcc_fts = np.asarray(mfcc_fts).T
print(np.shape(mfcc_fts))

from hmmlearn import hmm

model = hmm.GaussianHMM(n_components=100)
print(np.shape(mfcc_fts)[0])
model = model.fit(mfcc_fts, [np.shape(mfcc_fts)[0]])

mfcc_test = get_mfcc_features(data[len(data)//2:], samplerate)
print(model.score(np.asarray(mfcc_test).T))
file = "../wavs/pop/pop.00007.wav"
data, samplerate = sf.read(file)
mfcc_test = get_mfcc_features(data[len(data)//2:], samplerate)
print(model.score(np.asarray(mfcc_test).T))
file = "../wavs/pop/pop.00005.wav"
data, samplerate = sf.read(file)
mfcc_test = get_mfcc_features(data[len(data)//2:], samplerate)
print(model.score(np.asarray(mfcc_test).T))
file = "../wavs/pop/pop.00013.wav"
data, samplerate = sf.read(file)
mfcc_test = get_mfcc_features(data[len(data)//2:], samplerate)
print(model.score(np.asarray(mfcc_test).T))
file = "../wavs/pop/pop.00025.wav"
data, samplerate = sf.read(file)
mfcc_test = get_mfcc_features(data[len(data)//2:], samplerate)
print(model.score(np.asarray(mfcc_test).T))