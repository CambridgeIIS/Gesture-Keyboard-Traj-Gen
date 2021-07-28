
from functools import partial
import random
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import math
import preprocess


alphabet = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
    'y', 'z']


alphabet_ord = list(map(ord, alphabet))
alpha_to_num = defaultdict(int, list(map(reversed, enumerate(alphabet))))

def load_prepare_data_real(batch_size, max_x_length, max_c_length,data_type):
    print('start loading real data')

    # num = 800
    np_load_old = partial(np.load)
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    # data_type = 'real'
    input_data = np.load('{}_data.npy'.format(data_type))
    label_data = np.load('{}_label.npy'.format(data_type))
    input_data = np.array(input_data)
    label_data = np.array(label_data)
    # input_data = input_data[:num]s
    np.load = np_load_old

    tsteps_asii_= 20

    padded_input_data = []
    for i, v in enumerate(input_data):
        v = list(interpolate(v, len(label_data[i]), tsteps_asii=tsteps_asii_)[:max_x_length])

        while len(v) < max_x_length:

            # result.append(np.zeros((compoent_num, joints)))
            v.append(v[-1]) #padding
        padded_input_data.append(np.asarray(v))
        # residual = max_x_length - v.shape[0]
        # padding_array = np.zeros([int(residual), 2])
        # padded_input_data.append(
        #     np.concatenate([v, padding_array], axis=0))

    padded_input_data = np.array(padded_input_data)

    padded_label_data = []
    for _, v in enumerate(label_data):
        v = np.array(v)[:max_c_length]
        residual = max_c_length - v.shape[0]
        padding_array = np.zeros([int(residual)])
        padded_label_data.append(
            np.concatenate([v, padding_array], axis=0))
    padded_label_data = np.array(padded_label_data, dtype=np.int32)

    input_data = padded_input_data
    label_data = padded_label_data


    number_samples = len(input_data)

    logging.info('finish loading real data ')
    logging.info('stroke shape {}'.format(np.shape(input_data)))
    logging.info('text shape {}'.format(np.shape(label_data)))
    logging.info('number of real samples: %d'%(number_samples))


    # (3) create python generator
    while True:

        stroke_batch = np.zeros([batch_size, max_x_length, 2], dtype=np.float32)
        label_batch = np.zeros([batch_size, max_c_length], dtype=np.float32)

        for i in range(batch_size):
            # retrieve random samples from bucket of size batch_size
            sample_idx = random.randint(0, len(input_data) - 1)
            stroke_batch[i,:len(input_data[sample_idx]),:] = input_data[sample_idx]
            label_batch[i,:len( label_data[sample_idx])] = label_data[sample_idx]

        # stroke_batch = []
        # label_batch = []
        #
        # for i in range(batch_size):
        #     # retrieve random samples from bucket of size batch_size
        #     sample_idx = random.randint(0, len(input_data) - 1)
        #     stroke_batch.append(input_data[sample_idx])
        #     label_batch.append(label_data[sample_idx])

        yield (stroke_batch, label_batch)

def load_prepare_data_fake(batch_size, max_x_length, max_c_length):
    print('start loading fake data')

    input_data_ = []
    label_data_ = []

    number_samples = 0

    np_load_old = partial(np.load)

    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    data_type = 'fake'
    input_data = np.load('{}_data.npy'.format(data_type))
    label_data = np.load('{}_label.npy'.format(data_type))
    input_data = np.array(input_data)
    label_data = np.array(label_data)

    np.load = np_load_old

    for i in range(len(input_data)):
        if math.isnan(input_data[i][0][0]):
            continue
        input_data_.append(input_data[i])
        label_data_.append(label_data[i])

        number_samples += 1

    logging.info('finish loading fake data')
    logging.info('number of fake samples: %d'%(number_samples))


    # (3) create python generator
    while True:

        stroke_batch = np.zeros([batch_size, max_x_length, 2], dtype=np.float32)
        label_batch = np.zeros([batch_size, max_c_length], dtype=np.float32)

        for i in range(batch_size):
            # retrieve random samples from bucket of size batch_size
            sample_idx = random.randint(0, len(input_data_) - 1)
            stroke_batch[i,:len(input_data_[sample_idx]),:] = input_data_[sample_idx]
            label_batch[i,:len( label_data_[sample_idx])] = label_data_[sample_idx]

        yield (stroke_batch, label_batch)


alphabet = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
            'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
            'y', 'z']

alphabet_ord = list(map(ord, alphabet))
alpha_to_num = defaultdict(int, list(map(reversed, enumerate(alphabet))))


def load_str(batch_size):
    f = open('enron_sample_3words.txt', 'r')
    str_all = f.readlines()
    str_list_ = []
    str_list = random.sample(str_all, batch_size)
    for i in range(batch_size):
        str_list_.append(str_list[i][:-1])

    return str_list_


def convert2labels(str_list, max_c_length, batch_size):
    fake_labels = np.zeros([batch_size, max_c_length], dtype=np.int32)

    for i, phrase in enumerate(str_list):
        ascii_label = encode_ascii(phrase)[:max_c_length]
        fake_labels[i, :len(ascii_label)] = ascii_label
    fake_labels = np.array(fake_labels)
    return fake_labels


def key_convert2csv():
    keys = []
    x_pos = []
    y_pos = []
    f = open('holokeyboard.txt', 'r')
    str = f.readline()
    str = f.readline()
    while len(str) > 1:
        info = str[:-1].split(';')
        keys.append(info[0])
        x_pos.append(int(info[1]))
        y_pos.append(int(info[2]))

        str = f.readline()

    df = pd.DataFrame({
        'keys': keys,
        'x_pos': x_pos,
        'y_pos': y_pos,
    })

    df.to_csv('new_holokeyboard.csv')

    return df


def interpolate(stroke, len_ascii, tsteps_asii=20):
    xy_coords = stroke

    if len(stroke) > 3:
        f_x = interp1d(np.arange(len(stroke)), stroke[:, 0])
        f_y = interp1d(np.arange(len(stroke)), stroke[:, 1])

        xx = np.linspace(0, len(stroke) - 1, len_ascii * tsteps_asii)
        yy = np.linspace(0, len(stroke) - 1, len_ascii * tsteps_asii)

        x_new = f_x(xx)
        y_new = f_y(yy)

        xy_coords = np.hstack([x_new.reshape(-1, 1), y_new.reshape(-1, 1)])

    return xy_coords


def get_stroke_sequence_(stroke, max_x_length):
    # offsets = coords_to_offsets(stroke)
    offsets = normalize(offsets)
    offsets = offsets[:max_x_length]

    return offsets


def fake_generator(fake_str_list, max_x_length, batch_size):
    df = pd.read_csv("new_holokeyboard.csv", index_col='keys')
    offset_list = np.zeros([batch_size, max_x_length, 2], dtype=np.float32)
    scale = 0.01
    for i, fake_str in enumerate(fake_str_list):
        stroke = []
        for j in fake_str:
            stroke.append([df.loc[j][1], df.loc[j][2]])
        stroke_np = np.reshape(np.array(stroke) * scale, (-1, 2))

        stroke_fi = add_noise(interpolate_linear(stroke_np, len(fake_str), tsteps_asii=preprocess.tsteps_asii_))
        offset = get_stroke_sequence_(stroke_fi, max_x_length)
        offset_list[i, :len(offset), :] = offset
    return offset_list
    # return stroke_np[:,0], stroke_np[:,1], stroke_fi[:,0], stroke_fi[:,1]


def key_convert2csv():
    keys = []
    x_pos = []
    y_pos = []
    f = open('holokeyboard.txt', 'r')
    str = f.readline()
    str = f.readline()
    while len(str) > 1:
        info = str[:-1].split(';')
        keys.append(info[0])
        x_pos.append(int(info[1]))
        y_pos.append(int(info[2]))

        str = f.readline()

    df = pd.DataFrame({
        'keys': keys,
        'x_pos': x_pos,
        'y_pos': y_pos,
    })

    df.to_csv('new_holokeyboard.csv')

    return df


def interpolate_linear(stroke, len_ascii, tsteps_asii=20):
    xy_coords = stroke

    if len(stroke) > 3:
        f_x = interp1d(np.arange(len(stroke)), stroke[:, 0])
        f_y = interp1d(np.arange(len(stroke)), stroke[:, 1])

        xx = np.linspace(0, len(stroke) - 1, len_ascii * tsteps_asii)
        yy = np.linspace(0, len(stroke) - 1, len_ascii * tsteps_asii)

        x_new = f_x(xx)
        y_new = f_y(yy)

        xy_coords = np.hstack([x_new.reshape(-1, 1), y_new.reshape(-1, 1)])

    return xy_coords


def decode_ascii(label):
    num_to_alpha = defaultdict(list)
    for key, values in alpha_to_num.items():
        num_to_alpha[values].append(key)

    string = []
    for j in range(len(label)):
        string.append(''.join(np.squeeze(np.array([num_to_alpha[i] for i in label[j]]))))
    return string


def denoise(coords):
    """
    smoothing filter to mitigate some artifacts of the data collection
    """

    x_new = savgol_filter(coords[:, 0], 7, 3, mode='nearest')
    y_new = savgol_filter(coords[:, 1], 7, 3, mode='nearest')
    stroke = np.hstack([x_new.reshape(-1, 1), y_new.reshape(-1, 1)])

    return stroke


def add_noise(coords, scale=0.03):
    """
    adds gaussian noise to strokes
    """

    # coords = np.copy(coords)
    # coords[1:, :] += np.random.normal(loc=0.0, scale=scale, size=coords[1:, :].shape)

    return coords


def encode_ascii(ascii_string):
    """
    encodes ascii string to array of ints
    """
    return np.array(list(map(lambda x: alpha_to_num[x], ascii_string)) + [0])


def normalize(offsets):
    """
    normalizes strokes to median unit norm
    """
    offsets = np.copy(offsets)
    offsets[:, :2] /= np.median(np.linalg.norm(offsets[:, :2], axis=1))
    return offsets


def coords_to_offsets(coords):
    """
    convert from coordinates to offsets
    """
    offsets = coords[1:, :2] - coords[:-1, :2]
    offsets = np.concatenate([np.array([[0, 0]]), offsets], axis=0)
    # offsets = np.concatenate([coords[1:, :2] - coords[:-1, :2], coords[1:, 2:3]], axis=1)
    # offsets = np.concatenate([np.array([[0, 0, 0]]), offsets], axis=0)
    return offsets


def offsets_to_coords(offsets):
    """
    convert from offsets to coordinates
    """
    return np.cumsum(offsets[:, :2], axis=0)


def interpolate_(stroke, len_ascii, tsteps_asii=20):
    """
    interpolates strokes using cubic spline
    """

    xy_coords = stroke[:, :2]

    if len(stroke) > 3:
        f_x = interp1d(np.arange(len(stroke)), stroke[:, 0], kind='cubic')
        f_y = interp1d(np.arange(len(stroke)), stroke[:, 1], kind='cubic')

        xx = np.linspace(0, len(stroke) - 1, len_ascii * tsteps_asii)
        yy = np.linspace(0, len(stroke) - 1, len_ascii * tsteps_asii)

        x_new = f_x(xx)
        y_new = f_y(yy)

        xy_coords = np.hstack([x_new.reshape(-1, 1), y_new.reshape(-1, 1)])

    return xy_coords


def fake_generator_saver(max_c_length, max_x_length):
    size = 2000

    fake_str = load_str(size)

    x_label = convert2labels(fake_str, max_c_length, size)
    x = fake_generator(fake_str, max_x_length, size)

    x_label = np.array(x_label)
    x = np.array(x)

    print('saved {} data'.format(size))
    np.save("fake_data", x)
    np.save("fake_label", x_label)


if __name__ == "__main__":

    fake_generator_saver(preprocess.MAX_CHAR_LEN, preprocess.MAX_CHAR_LEN*preprocess.tsteps_asii_)
