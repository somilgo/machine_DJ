import librosa
import soundfile as sf
import numpy as np

STRIDE_SIZE = 10.
FRAME_SIZE = 25.
N_MFCC = 16

WINDOW = 3000
WINDOW_STRIDE = 30

def find_peaks(X, Y, iterations=1):
	out = list(zip(X, Y))
	for i in range(iterations):
		temp_out = []
		for j in range(len(out)):
			if (j-1) >= 0 and (j+1) < len(out):
				if out[j-1][1] < out[j][1] and out[j][1] < out[j+1][1]:
					temp_out.append(out[j])
		out = temp_out
	return out

def window_number_to_sec(win_num, win_len = WINDOW):
	stamp = win_num + (win_len // 2)
	return (FRAME_SIZE + STRIDE_SIZE * (stamp - 1)) / 1000.

def get_mfcc_features(song_data, sample_rate, stride_size = STRIDE_SIZE, frame_size = FRAME_SIZE):
	mfccs = librosa.feature.mfcc(song_data, sample_rate, 
								 n_mfcc=N_MFCC,
								 hop_length=int(STRIDE_SIZE / 1000. * sample_rate), 
								 n_fft=int(FRAME_SIZE / 1000. * sample_rate))
	return mfccs

# file = "../wavs/pop/pop.00009.wav"
# data, samplerate = sf.read(file)

# file = "../wavs/rock/rock.00021.wav"
# data2, samplerate2 = sf.read(file)
# print(len(data))
# joined_data = np.concatenate((data[200:200+len(data)//2], data2[len(data2)//2:]))
# print(np.shape(joined_data))

file = "../kiss.wav"
data, samplerate = sf.read(file)
	
print(np.shape(data))
data = data[:, 0]
mfccs = get_mfcc_features(data, samplerate)
mfccs = np.asarray(mfccs).T
print(np.shape(mfccs))

from hmmlearn import hmm

low_score = 1e9
low_score_index = -1

X = []
Y = []

for i in range(0, np.shape(mfccs)[0], WINDOW_STRIDE):
	if np.shape(mfccs)[0] - WINDOW <= i:
		break
	model = hmm.GaussianHMM(n_components=1	)
	model = model.fit(mfccs[i:i+WINDOW//2], [WINDOW//2])
	curr_score = model.score(mfccs[i+WINDOW//2:i+WINDOW], [len(mfccs[i+WINDOW//2:i+WINDOW])])
	print(curr_score, i, window_number_to_sec(i))
	X.append(window_number_to_sec(i))
	Y.append(curr_score * -1)
	if curr_score < low_score:
		low_score = curr_score
		low_score_index = i


peaks = find_peaks(X, Y, n=2)

import matplotlib.pyplot as plt  
plt.plot(X, Y)
for p in peaks:
	plt.axvline(x=p[0])
plt.show()

print(low_score, low_score_index)	