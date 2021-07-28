from __future__ import print_function
import os
from xml.etree import ElementTree
import csv
import numpy as np
import drawing
import glob


def get_stroke_sequence(fname):

    coords = []
    # fname='data/preprocessed_data_all/a_a_big_scratch_on_the_tabletop.csv'
    with open(fname, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            if row == []:
                continue
            row_split = row
            coords.append([float(row_split[0]),
                           float(row_split[1]),
                           int(0)
            ])
    coords = np.array(coords)
    offsets = drawing.coords_to_offsets(coords)
    offsets = offsets[:drawing.MAX_STROKE_LEN]
    offsets = drawing.normalize(offsets)
    return offsets


def get_ascii_sequences(phrases):
    lines = drawing.encode_ascii(phrases)[:drawing.MAX_CHAR_LEN]
    return lines


def collect_data():
    path = '../preprocess/preprocessed_data_all/*.csv'
    stroke_fnames=[]
    for fname in sorted(glob.glob(path), reverse=True):
        stroke_fnames.append(fname)
    return stroke_fnames


if __name__ == '__main__':
    asciis = []
    strokes = []

    print('traversing data directory...')
    stroke_fnames = collect_data()

    x = np.zeros([len(stroke_fnames), drawing.MAX_STROKE_LEN, 3], dtype=np.float32)
    x_len = np.zeros([len(stroke_fnames)], dtype=np.int16)
    c = np.zeros([len(stroke_fnames), drawing.MAX_CHAR_LEN], dtype=np.int8)
    c_len = np.zeros([len(stroke_fnames)], dtype=np.int8)

    path = '../preprocess/preprocessed_data_all/*.csv'
    for i, fname in enumerate(sorted(glob.glob(path), reverse=True)):
        phrases = fname.split('.')[2]
        phrases = phrases.split('_')[3:]
        phrases = ' '.join(phrases)
        print(phrases)
        x_i = get_stroke_sequence(fname)
        c_i = get_ascii_sequences(phrases)

        x[i, :len(x_i), :] = x_i
        x_len[i] = len(x_i)

        c[i, :len(c_i)] = c_i
        c_len[i] = len(c_i)

    np.save('data/processed/x.npy', x)
    np.save('data/processed/x_len.npy', x_len)
    np.save('data/processed/c.npy', c)
    np.save('data/processed/c_len.npy', c_len)
