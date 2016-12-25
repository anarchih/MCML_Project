# -*- coding: utf-8 -*-

import sys
import os
from scipy.io import wavfile
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import aubio
import audiotools
from audiotools import player


def tempo_detect(filename, method):
    samplerate = 44100
    # fft size
    win_s = 512
    # hop size
    hop_s = 256

    s = aubio.source(filename, samplerate, hop_s)
    samplerate = s.samplerate

    # default, complexdomain, hfc, phase, specdiff, energy, kl, mkl
    if method == '-b':
        o = aubio.tempo('complexdomain', win_s, hop_s)
    if method == '-o':
        o = aubio.onset('complexdomain', win_s, hop_s)
        # minimum inter onset interval [default=12ms]
        o.set_minioi_ms(20)
    # onset peak picking threshold [default=0.3]
    o.set_threshold(0.5)

    # list of beats, in samples
    beats = []
    total_frames = 0

    while True:
        samples, read = s()
        if o(samples):
            beats.append(o.get_last())
        total_frames += read
        if read < hop_s:
            break

    del s
    return beats


def read_dir(path):
    fileList = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.wav'):
                fileList.append(os.path.join(root, file))
    return fileList


def audio_play(filename):
    pList = [p.NAME for p in player.available_outputs()]
    slot = player.open_output(pList[0])
    test = player.Player(slot)
    test.open(audiotools.open(filename))
    test.play()


def output_result(filename, beats):
    rate, data = wavfile.read(filename)
    print(filename)
    length = len(data[:, 0])
    s = np.array(np.multiply(data, 0.2))
    d = 100
    count = 0
    for beat in beats:
        if (beat + d) < length:
            for i in range(beat - d, beat + d):
                s[i] = data[i]
            count += 1
        else:
            break

    minute = ((data.size) / 2 / (44100 * 60))
    bpm = '{:.2f}'.format(len(beats) / minute)
    print(bpm)
    path = 'output' + filename[6:len(filename) - 4] + '(' + str(bpm) + ').wav'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wavfile.write(path, rate, s)


def part_split(start, length, max_size):
    tmp = length
    count = 1
    while tmp >= max_size:
        tmp //= 2
        count *= 2
    split = []
    for i in range(1, count):
        split.append(start + tmp * i)
    return split


def audio_cut(filename, beats):
    rate, data = wavfile.read(filename)
    size = data.size / 2
    print(filename)
    cut = [0]
    max_size = 50000
    min_size = 10000
    for beat in beats:
        if beat != cut[-1] and beat <= size:
            length = beat - cut[-1]
            if length >= max_size:
                split = part_split(cut[-1], length, max_size)
                for s in split:
                    cut.append(s)
            if length >= min_size and length < max_size:
                cut.append(beat)
    cut.pop(0)
    length = size - cut[-1]
    if length >= max_size:
        split = part_split(cut[-1], length, max_size)
        for s in split:
            cut.append(s)
    parts = np.split(data, cut)
    if cut[-1] == size:
        parts.pop()

    minute = size / (44100 * 60)
    bpm = '{:.2f}'.format(len(cut) / minute)
    print(bpm)

    path = 'music' + filename[6:len(filename) - 4] + '(' + str(bpm) + ')/'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for i in range(len(parts)):
        wavfile.write(path + str(i) + '.wav', rate, parts[i])


def draw_plot(filename, beats, length=None):
    rate, data = wavfile.read(filename)
    mpl.rcParams['agg.path.chunksize'] = 20000
    fig, ax1 = plt.subplots()
    t = np.arange(0, length, 1)
    if length is None:
        length = len(data[:, 0])
    s1 = data[:, 0][:length]
    s2 = [0] * length
    count = 0
    for beat in beats:
        if beat < length:
            s2[beat] = s1[beat]
            count += 1
        else:
            break

    print(count)
    ax1.plot(t, s1, 'b-')
    ax1.set_xlabel('frames')
    ax1.set_ylabel('amp', color='b')
    ax2 = ax1.twinx()
    ax2 = ax1.twiny()
    ax2.plot(t, s2, 'r-')

    plt.show()


def main():
    option = sys.argv[1]
    output = sys.argv[2]
    method = sys.argv[3]
    path = sys.argv[4]
    if option == '-f':
        beats = tempo_detect(path, method)
        if output == '-t':
            output_result(path, beats)
        if output == '-c':
            audio_cut(path, beats)
    if option == '-d':
        fileList = read_dir(path)
        if output == '-t':
            for filename in fileList:
                beats = tempo_detect(filename, method)
                output_result(filename, beats)
        if output == '-c':
            for filename in fileList:
                beats = tempo_detect(filename, method)
                audio_cut(filename, beats)


if __name__ == '__main__':
    main()
