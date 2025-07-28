## Import libraries ##
import os
import pyedflib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.signal import butter, filtfilt
import wfdb

## Set working directory, adjust accordingly  ##
os.chdir("/Users/marcela/Downloads/abdominal-and-direct-fetal-ecg-database-1.0.0")

## Load EDF signal ##
edf = pyedflib.EdfReader('r01.edf')
n_channels = edf.signals_in_file
fs = int(edf.getSampleFrequency(0))
signals = np.zeros((edf.getNSamples()[0], n_channels))
for i in range(n_channels):
    signals[:, i] = edf.readSignal(i)
edf._close()

## Bandpass filter (1–40 Hz) ##
def bandpass_filter(signal, lowcut=1.0, highcut=40.0, fs=1000, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)

filtered_signals = np.apply_along_axis(bandpass_filter, axis=0, arr=signals, fs=fs)

## Standardize ##
signals_std = StandardScaler().fit_transform(filtered_signals)

## PCA decomposition ##
pca = PCA(n_components=n_channels)
pca_components = pca.fit_transform(signals_std)

## Plot first 2000 samples of all PCA components ##
plt.figure(figsize=(12, 6))
for i in range(pca_components.shape[1]):
    plt.plot(pca_components[:2000, i] + i * 10, label=f'PC {i+1}')
plt.title("PCA Components (first 2000 samples)")
plt.xlabel("Sample")
plt.ylabel("Amplitude (offset)")
plt.legend()
plt.tight_layout()
plt.show()

## Overlay fetal R-peaks on PC1 for comparison ##
annotation = wfdb.rdann('r01', extension='qrs')
r_peaks = annotation.sample

plt.figure(figsize=(12, 4))
plt.plot(pca_components[:, 0], label='PC1')
plt.scatter(r_peaks, pca_components[r_peaks, 0], color='red', label='Fetal R-peaks')
plt.title('PCA Component 1 with Fetal R-peaks')
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.show()
