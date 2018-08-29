# -*- coding: utf-8 -*-
import numpy as np
from scipy.signal import butter, lfilter
from pydub import AudioSegment
from utils import db_to_float, read_data_from_AudioSegment
from config import Config

# https://github.com/timsainb/python_spectrograms_and_inversion

# Most of the Spectrograms and Inversion are taken from: https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def overlap(X, window_size, window_step):
    """
    Create an overlapped version of X
    Parameters
    ----------
    X : ndarray, shape=(n_samples,)
        Input signal to window and overlap
    window_size : int
        Size of windows to take
    window_step : int
        Step size between windows
    Returns
    -------
    X_strided : shape=(n_windows, window_size)
        2D array of overlapped X
    """
    if window_size % 2 != 0:
        raise ValueError("Window size must be even!")
    # Make sure there are an even number of windows before stridetricks
    append = np.zeros((window_size - len(X) % window_size))
    X = np.hstack((X, append))

    ws = window_size
    ss = window_step
    a = X

    valid = len(a) - ws
    nw = (valid) // ss
    out = np.ndarray((nw, ws), dtype=a.dtype)

    for i in range(nw):
        # "slide" the window along the samples
        start = i * ss
        stop = start + ws
        out[i] = a[start: stop]

    return out


def stft(X, fftsize=128, step=65, mean_normalize=True, real=False,
         compute_onesided=True):
    """
    Compute STFT for 1D real valued input X
    """
    if real:
        local_fft = np.fft.rfft
        cut = -1
    else:
        local_fft = np.fft.fft
        cut = None
    if compute_onesided:
        cut = fftsize // 2
    if mean_normalize:
        X -= X.mean()

    X = overlap(X, fftsize, step)

    size = fftsize
    win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(size) / (size - 1))
    X = X * win[None]
    X = local_fft(X)[:, :cut]
    return X


def pretty_spectrogram(d, log=True, fft_size=512, step_size=64):
    """
    creates a spectrogram
    log: take the log of the spectrgram
    thresh: threshold minimum power for log spectrogram
    """
    specgram = np.abs(stft(d, fftsize=fft_size, step=step_size, real=False,
                           compute_onesided=True))

    if log == True:
        specgram /= specgram.max()  # volume normalize to max 1
        specgram = np.log10(specgram)  # take log
    return specgram




def create_spectrogram_per_second(data, mode='train'):
    # Only use a short clip for our demo
    step = Config.FRAME_RATE // 2 if mode == 'train' else Config.FRAME_RATE
    for i in range(0, len(data) - Config.FRAME_RATE, step):
        _data = data[i:i + Config.FRAME_RATE]
        if np.sqrt(np.sum([x ** 2 for x in _data])/Config.FRAME_RATE) < Config.SILENCE_THRESH:
            continue
        rt = pretty_spectrogram(butter_bandpass_filter(_data, Config.LOWCUT, Config.HIGHTCUT, Config.FRAME_RATE, order=1)
                                .astype('float64'),
                                fft_size=Config.FFT_SIZE,
                                step_size=Config.STEP_SIZE, log=True)
        if not np.isnan(rt).any():
            if rt.shape == (Config.IMG_W, Config.IMG_H):
                yield rt


if __name__ == '__main__':
    from utils import read_data, read_data_from_AudioSegment
    from pydub import AudioSegment
    myvideo = '/home/mindu/Desktop/ZaloChallenge/data/music/data/test/1/984537911385598887.mp3'
    audio = AudioSegment.from_file(myvideo)
    import matplotlib.pyplot as plt
    data = create_spectrogram_per_second(audio)
    for d in data:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 4))
        cax = ax.matshow(np.transpose(d), interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot,
                         origin='lower')
        fig.colorbar(cax)
        plt.title('Original Spectrogram')
        plt.show()


