## Importing relevant libraries ##
import os
import pyedflib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt
import wfdb


## Set your working directory ##
os.chdir("/Users/marcela/Downloads/abdominal-and-direct-fetal-ecg-database-1.0.0")


## Load in EDF file using pyedflib ##
edf = pyedflib.EdfReader('r01.edf')
n_channels = edf.signals_in_file
fs = int(edf.getSampleFrequency(0))  # assume all channels have same fs #

## Load all signal channels ##
signals = np.zeros((edf.getNSamples()[0], n_channels))
for i in range(n_channels):
    signals[:, i] = edf.readSignal(i)
edf._close()


## Bandpass filter (1–40 Hz) - Preprocessing step to ensure ##
def bandpass_filter(signal, lowcut=1.0, highcut=40.0, fs=1000, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, signal)

filtered_signals = np.apply_along_axis(bandpass_filter, axis=0, arr=signals, fs=fs)

## Standardize ##
signals_std = StandardScaler().fit_transform(filtered_signals)

## atuacl ICA ##
ica = FastICA(n_components=n_channels, random_state=42)
components = ica.fit_transform(signals_std)


## Plotting ICA components 1-5 ##

plt.figure(figsize=(12, 6))
for i in range(components.shape[1]):
    plt.plot(components[:2000, i] + i * 10, label=f'IC {i+1}')
plt.title("ICA Components (first 2000 samples)")
plt.xlabel("Sample")
plt.ylabel("Amplitude (offset)")
plt.legend()
plt.tight_layout()
plt.show()


## Load fetal R-peak annotations from WFDB ##

annotation = wfdb.rdann('r01', extension='qrs')
r_peaks = annotation.sample

## Plot ICA component with R-peaks ##
component_index = 0
plt.figure(figsize=(12, 4))
plt.plot(components[:, component_index], label=f'Component {component_index+1}')
plt.scatter(r_peaks, components[r_peaks, component_index], color='red', label='R-peaks')
plt.title(f'ICA Component {component_index+1} with R-peaks')
plt.xlabel("Sample")
plt.legend()
plt.tight_layout()
plt.show()



