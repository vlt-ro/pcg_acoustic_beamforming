import sys
import os
import glob
import pandas as pd
import numpy as np
import scipy.signal
sys.path.append('/Users/roquemev/Documents/OHSU/Research/TelluscopeProject/heart_sound_segmentation/tools')
from scipy import signal
import read_data
import quality_assessment
import peak_detection
import peak_classification
import segmentation
import librosa
from scipy import signal
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
import pywt

from scipy.io import loadmat
from options import *

def read_data(path):
    mat_file = loadmat(path)
    df_signal = pd.DataFrame(columns=keys)

    for k in keys:
        df_signal[k] = mat_file[k][init_time:final_time].reshape(-1)

    return df_signal

def filter_signals(df_signal,cutoff):
    df_signal_filter = pd.DataFrame(columns=keys)
    df_signal_filter['timestamps'] = df_signal['timestamps']

    sos_hs = signal.butter(N=6, Wn=cutoff, btype='bandpass', fs=Fs_original, output='sos')

    for k in keys[1:]:
        df_signal_filter[k] = signal.sosfiltfilt(sos_hs,df_signal[k])
    return df_signal_filter

def downsample_signal(df_signal,target_fs = Fs):
    df_signal_down = pd.DataFrame(columns=keys)

    for k in keys:
        dsignal = df_signal[k]

        if Fs_original==target_fs:
            df_signal_down[k] = dsignal

        downsampling_factor = Fs_original//target_fs

        if (downsampling_factor < 2) or (downsampling_factor > 12):
            return None
        else:
            signal_down = signal.decimate(dsignal,downsampling_factor)
            df_signal_down[k] = signal_down

    return  df_signal_down

def wavelet_denoise(signal,noise=None, wavelet='db4', level=6, k = 1.0):
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level, mode='constant')

    if isinstance(noise, np.ndarray):
        coeffs_noise = pywt.wavedec(noise, wavelet=wavelet, level=level, mode='constant')

    for ii in np.arange(level + 1):
        if isinstance(noise, np.ndarray):
            local_th = k*np.max(np.abs(coeffs_noise[ii]))
        else:
            local_th = k*np.std(coeffs[ii])

        coeffs[ii] = pywt.threshold(coeffs[ii], local_th, mode='soft')

    signal_denoised = pywt.waverec(coeffs,wavelet,mode='constant')
    if len(signal_denoised)>len(signal):
        signal_denoised = signal_denoised[:len(signal)]
    return signal_denoised

def noise_cancelling(df_signal,analysis_window):
    frame_len = int(analysis_window*Fs)
    df_signal_nonoise = pd.DataFrame(columns=['timestamps']+mic_denoise)
    df_signal_nonoise['timestamps'] = df_signal['timestamps'].values

    for channel in mic_denoise:
        new_signal = np.array([])
        for i in range(0, int(len(df_signal) / frame_len) + (len(df_signal) % frame_len > 0)):
            sig1 = df_signal.loc[i*frame_len:(i+1)*frame_len-1,channel].values

            if channel == 'microphone_bell':
                sig2 = df_signal.loc[i*frame_len:(i+1)*frame_len-1,'microphone_external'].values
                denoise = wavelet_denoise(sig1,sig2)
            else:
                denoise = wavelet_denoise(sig1)

            new_signal = np.concatenate((new_signal,denoise))

        df_signal_nonoise[channel] = new_signal

    return df_signal_nonoise

def rescaling(df_signal):
    df_signal_scaled = pd.DataFrame(columns=['timestamps']+mic_denoise)
    df_signal_scaled['timestamps'] = df_signal['timestamps'].values

    for channel in mic_denoise:
        df_signal_scaled[channel] = df_signal[channel]/np.max(np.abs(df_signal[channel]))

    return df_signal_scaled

def get_thresholds(df_signal,analysis_window,th):

    frame_len = int(analysis_window*Fs)

    width_info = np.array([])
    height_info = np.array([])
    sos_hs = signal.butter(N=6, Wn=200, btype='lowpass', fs=Fs_original, output='sos')

    for mic in mic_denoise:
        if mic!=reference_channel:
            for i in range(0, int(len(df_signal) / frame_len) + (len(df_signal) % frame_len > 0)):
                sig1 = df_signal.loc[i*frame_len:(i+1)*frame_len,reference_channel].values
                sig2 = df_signal.loc[i*frame_len:(i+1)*frame_len,mic].values

                xcorr = signal.correlate(sig1,sig2)
                #lags = signal.correlation_lags(len(sig2), len(sig1))

                analytic_signal = signal.hilbert(xcorr)
                amplitude_envelope = np.abs(analytic_signal)
                amplitude_envelope = signal.sosfiltfilt(sos_hs,amplitude_envelope)

                #plt.plot(xcorr)
                #plt.plot(amplitude_envelope)
                #plt.show()

                peaks,property = signal.find_peaks(amplitude_envelope,width=0)

                width_info = np.concatenate((width_info,np.array(property['widths'])))
                height_info = np.concatenate((height_info,amplitude_envelope[peaks]))

    th_height = np.percentile(height_info,th)
    th_width = np.percentile(width_info,th)
    return th_width,th_height,width_info,height_info

def optimal_delayed_signal(df_signal,analysis_window,th_width,th_height):

    df_signal_align = pd.DataFrame(columns=['timestamps']+mic_denoise)
    df_signal_align['timestamps'] = df_signal['timestamps'].values
    df_signal_align[reference_channel] = df_signal[reference_channel].values

    frame_len = int(analysis_window*Fs)
    n_frames = int(len(df_signal) / frame_len) + (len(df_signal) % frame_len > 0)
    sos_hs = signal.butter(N=6, Wn=200, btype='lowpass', fs=Fs_original, output='sos')

    for mic in mic_denoise:
        if mic!=reference_channel:
            new_signal = np.array([])

            for i in range(0, n_frames):
                sig1 = df_signal.loc[i*frame_len:(i+1)*frame_len-1,reference_channel].values
                sig2 = df_signal.loc[i*frame_len:(i+1)*frame_len-1,mic].values
                lags = signal.correlation_lags(len(sig2), len(sig1))

                xcorr = signal.correlate(sig1,sig2)
                analytic_signal = signal.hilbert(xcorr)
                amplitude_envelope = np.abs(analytic_signal)
                amplitude_envelope = signal.sosfiltfilt(sos_hs,amplitude_envelope)

                peaks,_ = signal.find_peaks(amplitude_envelope,width=th_width,height=th_height)

                # Remove
                if len(peaks)==0:
                    new_signal = np.concatenate((new_signal,np.zeros(len(sig2))))

                else:
                    ind = np.argsort(amplitude_envelope[peaks])
                    for j in reversed(range(len(ind))):
                        if np.abs(lags[peaks][j])<500:
                            delay = lags[peaks][j]
                            break
                    #ind = np.argmax(amplitude_envelope[peaks])
                    #delay = lags[peaks][ind]
                    try :
                        if delay>0:
                            if i==0 or i == n_frames-1:
                                s2_align = np.append(np.zeros(delay),sig2[:-delay])
                            else:
                                s2_align = df_signal.loc[i*frame_len+delay:(i+1)*frame_len-1+delay,mic].values
                        elif delay<0:
                            if i==n_frames-1:
                                s2_align = np.append(sig2[-delay:],np.zeros(-delay))
                            else:
                                s2_align = df_signal.loc[i*frame_len-delay:(i+1)*frame_len-1-delay,mic].values
                        elif delay==0:
                            s2_align = df_signal.loc[i*frame_len:(i+1)*frame_len-1,mic].values

                        new_signal = np.concatenate((new_signal,s2_align))

                    except UnboundLocalError:
                        new_signal = np.concatenate((new_signal,np.zeros(len(sig2))))

            df_signal_align[mic] = new_signal

    return  df_signal_align

def xcorr(df_signal, analysis_window):

    frame_len = int(analysis_window*Fs)
    df_xcorr = pd.DataFrame(columns=mic_denoise)

    M = len(mic_denoise)

    for key1 in mic_denoise:
        for i in range(0, int(len(df_signal) / frame_len) + (len(df_signal) % frame_len > 0)):
            xcorr = 0
            sig1 = df_signal.loc[i*frame_len:(i+1)*frame_len-1,key1].values
            for key2 in mic_denoise:
                if key2!=key1:
                    sig2 = df_signal.loc[i*frame_len:(i+1)*frame_len-1,key2].values
                    xcorr += np.max(signal.correlate(sig1,sig2))

            df_xcorr.loc[i,key1] = xcorr/(M-1)

    return df_xcorr

def weight_adaptation(df_signal,analysis_window,alpha):
    df_xcorr = xcorr(df_signal,analysis_window)

    M = len(mic_denoise)
    frame_len = int(analysis_window*Fs)

    df_weight_adaptation = pd.DataFrame(columns=mic_denoise)

    for i in range(0, int(len(df_signal) / frame_len) + (len(df_signal) % frame_len > 0)):
        for key1 in mic_denoise:
            if df_xcorr.loc[i,key1]<1/(4*M):
                df_weight_adaptation.loc[i, key1] = 0

            else:
                if i == 0:
                    df_weight_adaptation.loc[i, key1] = 1/M
                else:
                    df_weight_adaptation.loc[i, key1] = (1-alpha)*(df_weight_adaptation.loc[i-1, key1]) + alpha*df_xcorr.loc[i,key1]

    for index in df_weight_adaptation.index:
        if (df_weight_adaptation.loc[index] == 0).all():
            df_weight_adaptation.loc[index,reference_channel] = 1
    return df_weight_adaptation


def channels_sum(df_signal,analysis_window,df_weight_adaptation):

    frame_len = int(analysis_window*Fs)
    y = np.array([])
    for i in range(0, int(len(df_signal) / frame_len) + (len(df_signal) % frame_len > 0)):

        sub_signal = np.zeros(len(df_signal.loc[i*frame_len:(i+1)*frame_len-1,keys[0]]))
        for key1 in mic_denoise:
            weight_relative = df_weight_adaptation.loc[i,key1]

            sub_signal += weight_relative * df_signal.loc[i*frame_len:(i+1)*frame_len-1,key1].values

        y = np.concatenate((y,sub_signal/np.max(np.abs(sub_signal))))

    return y
    
