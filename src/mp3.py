import pydub
import numpy as np

def write(f, sr, x, normalized=False):
    """numpy array to MP3"""
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(x * 2 ** 15)
    else:
        y = np.int16(x*np.iinfo(np.int16).max)

    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(f, format="mp3",
        codec='mp3',
        bitrate='160000')


#write('pcg_valentina.mp3', Fs, filtered_signals[keys[1]])