import glob

import matplotlib.pyplot as plt
from options import *
from  src import visualizations, beamforming, mp3

## Read data
dir = r'data/'
file = glob.glob(dir + "*.mat")
file.sort()

index = 2
name = file[index].split('/')[-1].split('.')[0]
signals = beamforming.read_data(file[index])

## Display raw data
#visualizations.plot_signals(signals,name,show=True)

## A. Individual Channel Signal Enhacement:

# Filter data
df_signal = beamforming.filter_signals(signals,[20,700])

# Downsampling data
df_signal = beamforming.downsample_signal(df_signal)

# Noise Cancelling
analysis_window = 2
df_signal_denoised = beamforming.noise_cancelling(df_signal,analysis_window)
#visualizations.plot_signals_denoised(df_signal_denoised,df_signal,name,show=True)

df_signal_scaled = beamforming.rescaling(df_signal_denoised)
#visualizations.plot_signals(df_signal_scaled,name,show=True,use_all=False)

## B. Audio Information Extraction

# Thresholds computation for delay
analysis_window = 1.5
th_width,th_height,width_info,height_info = beamforming.get_thresholds(df_signal_scaled,analysis_window,90)
#visualizations.boxplot(th_width,th_height,width_info,height_info,True)

## C. TDOA Values Selection

# 1. Bad Quality Segments Elimination and Delay Signal
df_signal_delayed = beamforming.optimal_delayed_signal(df_signal_scaled,analysis_window,th_width,th_height)
#visualizations.plot_signals(df_signal_delayed,name,show=True,use_all=False)

## D. Output signal Generation

# 1. Interchannel Output Weight Adaptation
alpha = 0.05
df_weight = beamforming.weight_adaptation(df_signal_delayed,analysis_window,alpha)
# 2. Channels Sum

y = beamforming.channels_sum(df_signal_delayed,analysis_window,df_weight)

fig, axes = plt.subplots(figsize=(8,5),nrows=2, sharex=True)


axes[0].plot(df_signal[reference_channel])
axes[1].plot(y)
plt.show()
#mp3.write('pcg_valentina.mp3', Fs, df_signal[reference_channel])
#mp3.write('pcg_valentina_process.mp3', Fs, y)