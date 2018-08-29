from config import Config
from utils import read_wav_file
from spectrogram import create_spectrogram_per_second

import subprocess
import os
import operator
import csv

from keras.models import load_model
import numpy as np

model = load_model(Config.MODEL)

def predict(input_file):
    global model

    # convert mp3 to wav file
    command = "/usr/bin/ffmpeg -i '{}' -vn -acodec pcm_s16le -ac 1 -ar 44100 -f wav {}"
    subprocess.call(command.format(input_file, input_file[:-3]+'wav'), shell=True)

    _, audio = read_wav_file(input_file[:-3] + 'wav')
    os.remove(input_file[:-3] + 'wav')
    d = {i:0 for i in range(1,11,1)}
    for _input in create_spectrogram_per_second(audio, mode='test'):
        y_prob = model.predict(np.expand_dims([_input], 3))
        y_classes = y_prob.argmax(axis=-1)[0] + 1
        d[y_classes] += 1
    return max(d.items(), key=operator.itemgetter(1))[0]





with open('test.csv', 'w', newline='') as csvfile:
    fieldnames = ['Id', 'Genre']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    with open(Config.TEST_CSV) as f:
        for line in f.readlines():
            filename = line.strip()
            writer.writerow({'Id': filename, 'Genre': predict(os.path.join(
                Config.RAW_TEST_DATA, filename
            ))})

