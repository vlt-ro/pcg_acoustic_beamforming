from options import *
import numpy as np
import matplotlib.pyplot as plt

def plot_signals(signals,name,show, use_all=True,save=False):
    if use_all:
        key_plot = keys[1:]
    else:
        key_plot = mic_denoise

    fig, axes = plt.subplots(figsize=(7,7),nrows=len(key_plot), sharex=True)
    plt.suptitle('Signal: '+name)
    for i,k in enumerate(key_plot):
        axes[i].plot(signals[keys[0]],signals[k],label = k)
        axes[i].legend(loc='upper right')

    plt.xlabel('Time [s]')
    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.close()

def plot_signals_denoised(signals,signals_before,name,show, save=False):

    fig, axes = plt.subplots(figsize=(7,7),nrows=len(mic_denoise), sharex=True)
    plt.suptitle('Signal: '+name)
    for i,k in enumerate(mic_denoise):
        axes[i].plot(signals_before[keys[0]],signals_before[k],label = k)

        axes[i].plot(signals[keys[0]],signals[k],label = k)
        axes[i].legend(loc='upper right')

    plt.xlabel('Time [s]')
    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.close()

def boxplot(th_width,th_height,width_info,height_info,show):

    plt.subplot(121)
    plt.boxplot(width_info)
    plt.axhline(th_width)
    plt.title('Peak Width')
    plt.subplot(122)
    plt.boxplot(height_info)
    plt.axhline(th_height)
    plt.title('Peak Height')

    if show:
        plt.show()
    else:
        plt.close()