from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import glob
import csv
import numpy as np
import data_utils
import collections


# path= 'processed_real_all/*.csv'
# path = 'holo_real/*.csv'
# path = 'processed_artificial_r/*.csv'
path = 'shawn/*.log'


def add_features(sequence):
    sequence = np.asarray(sequence)
    next_seq = np.append(sequence[1:, :], [sequence[-1, :]], axis=0)
    prev_seq = np.append([sequence[0, :]], sequence[:-1, :], axis=0)

    # compute gradient
    gradient = np.subtract(sequence, prev_seq)

    #compute curvature
    vec_1 = np.multiply(gradient, -1)
    vec_2 = np.subtract(next_seq, sequence)
    angle = np.divide(np.sum(vec_1*vec_2, axis=1),
                      np.linalg.norm(vec_1, 2, axis=1)*np.linalg.norm(vec_2, 2, axis=1))

    angle[np.isnan(angle)]=0

    curvature = np.column_stack((np.cos(angle), np.sin(angle)))

    #compute vicinity (5-points) - curliness/linearity
    padded_seq = np.concatenate(([sequence[0]], [sequence[0]], sequence, [sequence[-1]], [sequence[-1]]), axis=0)
    aspect = np.zeros(len(sequence))
    slope = np.zeros((len(sequence), 2))
    curliness = np.zeros(len(sequence))
    linearity = np.zeros(len(sequence))
    for j in range(2, len(sequence)+2):
        vicinity = np.asarray([padded_seq[j-2], padded_seq[j-1], padded_seq[j], padded_seq[j+1], padded_seq[j+2]])
        delta_x = max(vicinity[:, 0]) - min(vicinity[:, 0])
        delta_y = max(vicinity[:, 1]) - min(vicinity[:, 1])

        # delta_x = vicinity[-1, 0] - vicinity[0, 0]
        # delta_y = vicinity[-1, 1] - vicinity[0, 1]
        slope_vec = vicinity[-1] - vicinity[0]

        #aspect of trajectory
        aspect[j-2] = (delta_y - delta_x) / (delta_y + delta_x)

        #cos and sin of slope_angle of straight line from vicinity[0] to vicinity[-1]
        slope_angle = np.arctan(np.abs(np.divide(slope_vec[1], slope_vec[0]))) * np.sign(np.divide(slope_vec[1], slope_vec[0]))
        slope[j-2] = [np.cos(slope_angle), np.sin(slope_angle)]

        #length of trajectory divided by max(delta_x, delta_y)
        curliness[j-2] = np.sum([np.linalg.norm(vicinity[k+1] - vicinity[k], 2) for k in range(len(vicinity)-1)]) / max(delta_x, delta_y)

        #avg squared distance from each point to straight line from vicinity[0] to vicinity[-1]
        linearity[j-2] = np.mean([np.power(np.divide(np.cross(slope_vec, vicinity[0] - point), np.linalg.norm(slope_vec, 1)), 2) for point in vicinity])

    vicinity_features = np.column_stack((aspect, slope, curliness, linearity))

    # add features to signal
    offsets = data_utils.coords_to_offsets(sequence)

    result = np.nan_to_num(np.concatenate((offsets, gradient, curvature, vicinity_features), axis=1)).tolist()

    return result





def get_ascii_sequences(phrases):
    lines = data_utils.encode_ascii(phrases)
    # lines = encode_ascii(phrases)
    return lines




def get_stroke_sequence(fname,phrase=None):

    coords = []

    with open(fname, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:

            if row[0] == '':
                continue
            if row == []:
                continue
            coords.append([float(row[0]),
                           float(row[1]),
            ])
    coords = np.array(coords)
    coords = np.reshape(coords, [-1, 2])

    coords = data_utils.denoise(coords)
    coords = data_utils.interpolate(coords, len(phrase) * 100)
    coords = data_utils.interpolate(coords, len(phrase))
    # coords = data_utils.interpolate(coords, len(phrase), tsteps_asii= tsteps_asii_)
    # offsets = data_utils.coords_to_offsets(coords)
    # offsets = data_utils.normalize(offsets)

    coords = data_utils.normalize(coords)
    return coords


def collect_data():

    stroke_fnames=[]

    for fname in sorted(glob.glob(path), reverse=True):
        stroke_fnames.append(fname)
    return stroke_fnames

def holo_main():
    text_line_data_all = []

    label_text_line_all = []

    char_len = []
    stroke_len = []
    open('shawn_word_holo.txt', 'w').close()

    last_word = ''
    word_list = []
    with open('shawn_word_holo.txt', 'w') as f:
        for i, fname in enumerate(sorted(glob.glob(path), reverse=False)):
            ############# label data #############
            print(fname)

            phrases = (fname.split('.')[-2]).split('/')[-1]
            phrases = phrases.split('_')[1:2]
            phrases = '_'.join(phrases)
            sequence = get_stroke_sequence(fname, phrases)


            phrases_ = phrases.split('_')
            phrases_ = ' '.join(phrases_)
            f.write("%s \n" % phrases_)

            text_line_data = sequence

            label_text_line = get_ascii_sequences(phrases)


            if len(text_line_data)==0:
                continue

            # if last_word!=phrases:
            #     print(phrases)
            #     print(get_ascii_sequences(phrases))
            #     last_word = phrases
            print(phrases)
            print(get_ascii_sequences(phrases))
            word_list.append(phrases)

            text_line_data_all.append(text_line_data)
            label_text_line_all.append(label_text_line)

            char_len.append(len(label_text_line))
            stroke_len.append(len(text_line_data))


    d = collections.OrderedDict()

    for i, v in enumerate(word_list):
        d[v] = i

    index_list = np.array(list(d.values()))
    print(index_list)
    text_line_data_all = [text_line_data_all[i] for i in index_list]
    label_text_line_all = [label_text_line_all[i] for i in index_list]
    char_len = [char_len[i] for i in index_list]
    stroke_len = [stroke_len[i] for i in index_list]


    text_line_data_all = np.array(text_line_data_all, dtype = 'object')
    label_text_line_all = np.array(label_text_line_all, dtype = 'object')

    # save as .npy
    np.save("real_data", text_line_data_all)
    np.save("real_label", label_text_line_all)
    print("Successfully saved!")
    print('average phrase length {}'.format(np.mean(char_len)))
    print('max phrase length {}'.format(np.max(char_len)))
    print('min phrase length {}'.format(np.min(char_len)))
    print('average stroke length {}'.format(np.mean(stroke_len)))
    print('max stroke length {}'.format(np.max(stroke_len)))
    print('min stroke length {}'.format(np.min(stroke_len)))
    print('shape {}'.format(np.shape(text_line_data_all)))

def main():
    text_line_data_all = []

    label_text_line_all = []

    # stroke_fnames = collect_data()
    # text_line_data_all = np.zeros([600, MAX_STROKE_LEN, 2], dtype=np.float32)
    # label_text_line_all = np.zeros([600, MAX_CHAR_LEN], dtype=np.int32)

    char_len = []
    stroke_len = []
    for i, fname in enumerate(sorted(glob.glob(path))):
        print(i)
        # if i%4 == 0:
        #     continue
        ############# label data #############
        print(fname)

        # phrases = (fname.split('.')[-2]).split('/')[-1]
        # phrases = phrases.split('_')[1:]
        # phrases = ' '.join(phrases)
        # sequence = get_stroke_sequence(fname, phrases)


        phrases = (fname.split('.')[-2]).split('/')[-1]
        phrases = phrases.split('_')[1:2]
        phrases = ' '.join(phrases)
        sequence = get_stroke_sequence(fname, phrases)

        print(phrases)

        text_line_data=sequence

        label_text_line = get_ascii_sequences(phrases)


        text_line_data_all.append(text_line_data)
        label_text_line_all.append(label_text_line)

        char_len.append(len(label_text_line))
        stroke_len.append(len(text_line_data))

    text_line_data_all = np.array(text_line_data_all, dtype = 'object')
    label_text_line_all = np.array(label_text_line_all, dtype = 'object')

    # save as .npy
    np.save("real_data", text_line_data_all)
    np.save("real_label", label_text_line_all)
    print("Successfully saved!")
    print('average phrase length {}'.format(np.mean(char_len)))
    print('max phrase length {}'.format(np.max(char_len)))
    print('min phrase length {}'.format(np.min(char_len)))
    print('average stroke length {}'.format(np.mean(stroke_len)))
    print('max stroke length {}'.format(np.max(stroke_len)))
    print('min stroke length {}'.format(np.min(stroke_len)))
    print('shape {}'.format(np.shape(text_line_data_all)))

# average phrase length 21.674291938997822
# max phrase length 41
# min phrase length 6
# average stroke length 459.5
# max stroke length 918
# min stroke length 1

if __name__ == "__main__":
    holo_main()
