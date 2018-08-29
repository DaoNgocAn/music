# -*- coding: utf-8 -*-
import wave

import numpy as np
from pydub import AudioSegment

def read_data_from_AudioSegment(AS):
    import numpy as np
    dtype = '<{}{}'.format(AS,AS.sample_width)
    return AS.frame_rate, np.frombuffer(AS.raw_data, dtype=dtype).reshape(-1, AS.channels)[:,0]

def read_data(file_name):
    file_ext = file_name.split('.')[-1]
    if file_ext == 'mp4':
        return read_data_from_AudioSegment(AudioSegment.from_file(file_name, file_ext))
    if file_ext == 'wav':
        return read_data_from_AudioSegment(AudioSegment.from_wav(file_name))
    if file_ext == 'mp3':
        return read_data_from_AudioSegment(AudioSegment.from_mp3(file_name))
    if file_ext == 'ogg':
        return read_data_from_AudioSegment(AudioSegment.from_ogg(file_name))
    if file_ext == 'flv':
        return read_data_from_AudioSegment(AudioSegment.from_flv(file_name))
    return read_data_from_AudioSegment(AudioSegment.from_file(file_name, file_ext))


def db_to_float(db, using_amplitude=True):
    """
    Converts the input db to a float, which represents the equivalent
    ratio in power.
    """
    db = float(db)
    if using_amplitude:
        return 10 ** (db / 20)
    else:  # using power
        return 10 ** (db / 10)

def read_wav_file(filename):
    w = wave.open(filename)
    dtype = '<i{}'.format(w.getsampwidth())
    r = w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), dtype=dtype).reshape(-1, w.getnchannels())[:, 0]
    w.close()
    return r
