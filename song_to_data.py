# -*- coding: utf-8 -*-
import os
import numpy as np
from shutil import copy2
import random

import spectrogram
from utils import read_data, read_data_from_AudioSegment, read_wav_file
from config import Config


from pydub import AudioSegment


def generate_label(train_size):
    if np.random.random() < train_size:
        return 0
    else:
        return 1

def mp3_to_wav():
    import subprocess
    import os
    command = "/usr/bin/ffmpeg -i '{}' -vn -acodec pcm_s16le -ac 1 -ar 44100 -f wav {}"

    for folder in ['train', 'test']:
        for label in os.listdir(os.path.join(Config.DATA_MP3, folder)):
            if not os.path.exists(os.path.join(Config.DATA_WAV, folder, label)):
                os.makedirs(os.path.join(Config.DATA_WAV, folder, label))
            for filename in os.listdir(os.path.join(Config.DATA_MP3, folder, label)):
                subprocess.call(command.format(os.path.join(Config.DATA_MP3, folder, label, filename),
                                               os.path.join(Config.DATA_WAV, folder, label, filename)[:-3] + 'wav'),
                                shell=True)

def split_train_test_data():
    if Config.NAME_LOCAL_MACHINE not in os.path.dirname(os.path.realpath(__file__)):
        pass
    for i in range(1,11,1):
        os.makedirs(Config.TRAIN_DATA_MP3 + '/{}'.format(i))
        os.makedirs(Config.TEST_DATA_MP3 + '/{}'.format(i))

    d_train ={i:[] for i in range(1,11,1)}
    d_test = {i:[] for i in range(1,11,1)}
    with open(Config.TRAIN_CSV) as f:
        for line in f.readlines():
            filename, label = line.strip().split(',')
            l = generate_label(0.8)
            if l == 0:
                d_train[int(label)].append(filename)
            else:
                d_test[int(label)].append(filename)
            # if l == 0:
            #     copy2(os.path.join(Config.RAW_DATA, filename),
            #           os.path.join(Config.TRAIN_DATA_MP3 + '/{}'.format(label), filename))
            # else:
            #     copy2(os.path.join(Config.RAW_DATA, filename),
            #           os.path.join(Config.TEST_DATA_MP3 + '/{}'.format(label), filename))
    for i in range(1,11,1):
        if len(d_train[i]) > Config.NB_SAMPLES:
            random.shuffle(d_train[i])
            d_test[i].extend(d_train[i][Config.NB_SAMPLES:])
            del d_train[i][Config.NB_SAMPLES:]
            for filename in d_train[i]:
                copy2(os.path.join(Config.RAW_DATA, filename),
                      os.path.join(Config.TRAIN_DATA_MP3 + '/{}'.format(i), filename))
            for filename in d_test[i]:
                copy2(os.path.join(Config.RAW_DATA, filename),
                      os.path.join(Config.TEST_DATA_MP3 + '/{}'.format(i), filename))
        else:
            times = Config.NB_SAMPLES // len(d_train[i])
            r = Config.NB_SAMPLES % len(d_train[i])
            for j in range(times):
                for filename in d_train[i]:
                    copy2(os.path.join(Config.RAW_DATA, filename),
                          os.path.join(Config.TRAIN_DATA_MP3 + '/{}'.format(i),
                                       filename[:-4] + '_{}'.format(j) + filename[-4:]))
            random.shuffle(d_train[i])
            for filename in d_train[i][:r]:
                copy2(os.path.join(Config.RAW_DATA, filename),
                      os.path.join(Config.TRAIN_DATA_MP3 + '/{}'.format(i),
                                   filename[:-4] + '_{}'.format(times) + filename[-4:]))
            for filename in d_test[i]:
                copy2(os.path.join(Config.RAW_DATA, filename),
                      os.path.join(Config.TEST_DATA_MP3 + '/{}'.format(i), filename))


def save_npy_slices():
    for folder in ['train', 'test']:
        for label in os.listdir(os.path.join(Config.DATA_WAV, folder)):
            if not os.path.exists(os.path.join(Config.NPY_SLICES_PATH, folder, label)):
                os.makedirs(os.path.join(Config.NPY_SLICES_PATH, folder, label))
            for filename in os.listdir(os.path.join(Config.DATA_WAV, folder, label)):
                try:
                    _, audio = read_wav_file(os.path.join(Config.DATA_WAV, folder, label, filename))
                except:
                    continue
                for i, data in enumerate(spectrogram.create_spectrogram_per_second(audio)):
                    try:
                        np.save(os.path.join(Config.NPY_SLICES_PATH, folder, label, filename[:-4]+'_{}.npy'.format(i)),
                                data)
                    except:
                        continue

if __name__ == '__main__':
    #split_train_test_data()
    #mp3_to_wav()
    save_npy_slices()
