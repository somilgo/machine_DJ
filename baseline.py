import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
import scipy.fftpack
import sunau
from pydub import AudioSegment
import soundfile as sf
import os
import pickle
	from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import librosa
from adversary.feature_extraction import get_mfcc_features


def get_wav(au_file):
	wav_file = au_file.replace(".au", ".wav")
	if not os.path.exists("wavs/" + wav_file):
		song = AudioSegment.from_file(au_file,'au')
		directory = "wavs/" + "/".join(au_file.split("/")[:-1])
		if not os.path.exists(directory):
			os.makedirs(directory)
		song.export("wavs/" + wav_file,format='wav')
	return "wavs/" + wav_file

def get_amps(data, samplerate):
	fourier = np.fft.fft(data)
	amps = abs(fourier[:(len(fourier)//2)])
	return amps

def get_freqs(samplerate, data_length):
	k = np.arange(data_length//2)
	T = data_length/samplerate  # where fs is the sampling frequency
	frqs = k/T

	return frqs

def get_tempo(file):
	seq, samplerate = sf.read(file)
	bpm = librosa.beat.tempo(seq, sr = samplerate)
	return bpm

def list_files(dir):
	r = []
	for root, dirs, files in os.walk(dir):
		for name in files:
			if name.endswith(".au"):
				r.append(root + "/" + name)
	return r

def get_sample_freqs(file, frame_size):
	seq, samplerate = sf.read(file) 
	samples = (seq[pos:pos + frame_size] for pos in range(0, len(seq), frame_size//10))
	out = []
	for s in samples:
		if (len(s) < frame_size):
			continue
		amps = get_amps(s, samplerate)
		out.append(get_amps(s, samplerate))
	return np.average(np.array(out), axis=0), samplerate, len(seq)

def purity(Xlabels, Klabels):
	res = {}
	for l in range(len(Xlabels)):
		label = Xlabels[l].split(".")[0]
		klabel = Klabels[l]
		if not res.get(klabel):
			res[klabel] = {}
		if not res[klabel].get(label):
			res[klabel][label] = 0
		res[klabel][label] += 1
	max_vals = 0
	for k in res:
		max_vals += max(res[k].values())

	return float(max_vals) / float(len(Xlabels))

def binning(fourier, freq, top, gap):
	freq_space = freq[1]
	fourier_integ = 0.
	thresh = gap
	bins = []
	for f in range(len(freq)):
		if freq[f] > top:
			break
		if freq[f] > thresh:
			gap *= 1.
			bins.append(fourier_integ)
			fourier_integ = 0.
			thresh += gap
		fourier_integ += fourier[f]
	return bins

def kmeans(Xbins, n=10):	
	# comps = PCA(n_components=6)
	# Xbins = comps.fit_transform(Xbins)
	kmeans = KMeans(n_clusters=n, n_init=40).fit(Xbins)

	klabels = kmeans.labels_
	res = {}
	for l in range(len(Xlabels)):
		label = Xlabels[l].split(".")[0]
		klabel = klabels[l]
		if not res.get(label):
			res[label] = [0] * 10
		res[label][klabel] += 1
	return klabels

# au_files = list_files("./")
# for au_file in au_files:
# 	get_wav(au_file)

file = "./wavs/hiphop/hiphop.00002.wav"
root_directory = "./wavs/"
sample_size = 2**13	
dir_num = 1
freq_cap = 1000
bin_size = 10
genre_count = 5
single = True

Xbins = []
mfccs = []
Xlabels = []

use_pickle = False
if use_pickle:
	_, samplerate, _ = get_sample_freqs(file, sample_size)
	frqs = get_freqs(samplerate, sample_size)
	fouriers = pickle.load(open("results/" + str(sample_size) + '.txt','rb'))
	Xlabels = pickle.load(open("results/" + str(sample_size) + "labels" + '.txt','rb'))

	if single:
		for f in fouriers:
			bins = binning(f, frqs, freq_cap, bin_size)
			Xbins.append(bins)

		klabels = kmeans(Xbins, n = genre_count)
		print("Freq Cap:", freq_cap, "Bin Size:", bin_size, "Purity", purity(Xlabels, klabels))
	else:
		for freq_cap in range(100, 10000, 100):
			for bin_size in range(5, 1000, 5):
				Xbins = []
				for f in fouriers:
					Xbins.append(binning(f, frqs, freq_cap, bin_size))
				try:
					klabels = kmeans(Xbins)
				except:
					break
			


else :
	fouriers = []
	for directory in next(os.walk(root_directory))[1][:genre_count]:
		
		dir_bins = []
		beat_tot = []

		for filename in os.listdir(directory)[:50]:

			fourier, samplerate, data_length = get_sample_freqs(directory + "/" + filename, sample_size)
			fouriers.append(fourier)
			frqs = get_freqs(samplerate, sample_size)
			bins = binning(fourier, frqs, freq_cap, bin_size)
			#bins = bins / np.linalg.norm(bins)
			dir_bins.append(bins)
			Xbins.append(bins)
			data, samplerate = sf.read(directory + "/" + filename)
			mfcc = get_mfcc_features(data, samplerate)
			mfcc_np = npa = np.asarray(mfcc, dtype=np.float32)
			mfcc_np_ave = np.mean(mfcc_np, axis=1)
			mfccs.append(mfcc_np_ave)
			Xlabels.append(filename)
			#beat_tot.append(get_tempo(directory + "/" + filename))

		plt.figure(dir_num)
		plt.title(directory)
		for b in dir_bins:
			plt.plot(range(len(b)), b, c=np.random.rand(3,))
		print("Done with", directory)
		dir_num += 1

if not use_pickle:
	pkl_file= open("results/" + str(sample_size) + '.txt','wb')
	pickle.dump(fouriers,pkl_file)

	pkl_file= open("results/" + str(sample_size) + "labels" + '.txt','wb')
	pickle.dump(Xlabels,pkl_file)

klabels = kmeans(mfccs, n = genre_count)
print(purity(Xlabels, klabels))

klabels = kmeans(Xbins, n = genre_count)
print(purity(Xlabels, klabels))

if not use_pickle:
	plt.show()