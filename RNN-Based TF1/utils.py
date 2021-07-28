import numpy as np
import math
import random
import os
import pickle as pickle
import xml.etree.ElementTree as ET
from sklearn import preprocessing


import csv
import numpy as np
import matplotlib.pyplot as plt
import glob
import os


class DataLoader():
    def __init__(self, args, logger, limit = 500):
        self.data_dir = args.data_dir
        self.threshold=args.threshold
        self.alphabet = args.alphabet
        self.scale = args.scale
        self.batch_size = args.batch_size
        self.tsteps = args.tstepshh
        self.data_scale = args.data_scale
        self.ascii_steps = args.tsteps/args.tsteps_per_ascii
        self.logger = logger
        self.limit = limit 

        data_file = os.path.join(self.data_dir, "strokes_training_data.cpkl")
        stroke_dir = self.data_dir + "/lineStrokes"
        ascii_dir = self.data_dir + "/ascii"

        if not (os.path.exists(data_file)) :
            self.logger.write("\tcreating training data cpkl file from raw source")
            self.preprocess(data_file)

        self.load_preprocessed(data_file)
        self.reset_batch_pointer()

    def preprocess(self,data_file):
        self.logger.write("\tparsing dataset...")

        asciis = []
        strokes = []
        for ID in range(21):
            path = '../Trace_data_processed\ID%s\*.csv' % (ID + 1)

            for fname in sorted(glob.glob(path), reverse=True):
                phrases = fname.split('_')[3:-2]

                phrases_join_ = ' '.join(phrases)

                if ID ==21:
                    phrases_join=phrases_join_

                    pos_list = []
                    pre_x = 0
                    pre_y = 0
                    with open(fname, newline='') as csvfile:
                        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                        for row in spamreader:

                            if row == []:
                                continue
                            row_split = row
                            pos_list.append([self.scale * (float(row_split[0]) - pre_x), self.scale * (float(row_split[1]) - pre_y),0])
                            pre_x = float(row_split[0])
                            pre_y = float(row_split[1])

                else:
                    phrases_join = phrases_join_+'_'
                    pos_list = []
                    pre_x=0
                    pre_y=0

                    with open(fname, newline='') as csvfile:
                        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                        for row in spamreader:
                            if row == []:
                                continue
                            row_split=row
                            pos_list.append([self.scale*(-float(row_split[0])-pre_x), self.scale*(float(row_split[1])-pre_y), 0])
                            pre_x=-float(row_split[0])
                            pre_y=float(row_split[1])

                counter_sum+=counter
                asciis.append(phrases_join)
                strokes.append(pos_list)

        assert (len(strokes) == len(asciis)), "There should be a 1:1 correspondence between stroke data and ascii labels."
        f = open(data_file, "wb")
        pickle.dump([strokes, asciis], f, protocol=2)
        f.close()

        self.logger.write("\tfinished parsing dataset. saved {} lines".format(len(strokes)))


    def load_preprocessed(self, data_file):
        f = open(data_file,"rb")
        [self.raw_stroke_data, self.raw_ascii_data] = pickle.load(f, encoding='latin1')

        f.close()


        self.stroke_data = []
        self.ascii_data = []
        self.valid_stroke_data = []
        self.valid_ascii_data = []

        cur_data_counter = 0
        zero = 0

        original_stroke_data = []
        original_ascii_data = []

        for m in range(len(self.raw_stroke_data)):
            data = self.raw_stroke_data[m]
            data=np.array(data)
            assert not np.any(np.isnan(data))

            ascii= self.raw_ascii_data[m]
            ascii_len = len(self.raw_ascii_data[m])
            average_tsteps_ = ascii_len * 12

            if len(data) < average_tsteps_:

                multiple = int(round(average_tsteps_ / len(data)))
                length = multiple * len(data)
                new_data = np.zeros((length, 3))

                for i in range(length):
                    if i % multiple is 0:
                        step = np.divide(data[int(i / multiple)], multiple)
                        step=np.reshape(step,(1,3))

                        for j in range(multiple):
                            new_data[(i + j),:] = step
                new_data=np.array(new_data)
                original_stroke_data.append(new_data)
                original_ascii_data.append(ascii)
            else:
                multiple = int(round(len(data) / average_tsteps_))
                length = int(round(len(data) / multiple))
                new_data = np.zeros((length + 1, 3))
                for ii in range(len(data)):

                    if ii % multiple is 0:
                        sum = np.sum(data[ii:ii + multiple],axis=0)
                        sum=np.reshape(sum,(1,3))

                        new_data[int(ii / multiple),:] = sum

                new_data=np.array(new_data)
                original_stroke_data.append(new_data)
                original_ascii_data.append(ascii)
        f = open("data/new_data", "wb")
        pickle.dump([original_stroke_data,original_ascii_data], f, protocol=2)
        f.close()

        original_stroke_data_ = []
        original_ascii_data_ = []
        tsteps = 300
        for i in range(len(original_stroke_data)):
            data = original_stroke_data[i]
            ascii = original_ascii_data[i]

            if len(data) < tsteps:
                multiple = int(round(tsteps / len(data)))
                length = multiple * len(data)
                new_data = []
                counter = 0
                for i in range(length):

                    if i % multiple is 0:
                        new_data.append(data[i - counter])
                    else:
                        new_data.append([0., 0., 0.])
                        counter += 1

                original_stroke_data_.append(new_data)
                original_ascii_data_.append(ascii)

            else:
                original_stroke_data_.append(data)
                original_ascii_data_.append(ascii)

        original_stroke_data = original_stroke_data_
        original_ascii_data = original_ascii_data_

        original_stroke_data_ = []
        original_ascii_data_ = []
        for i in range(len(original_stroke_data)):
            data = original_stroke_data[i]
            ascii = original_ascii_data[i]

            if len(data) < tsteps:
                pad = int(tsteps - np.shape(data)[0])
                data_ = np.concatenate((np.zeros((pad, 3)), data), axis=0)
                original_stroke_data_.append(data_)
                original_ascii_data_.append(ascii)
            else:
                original_stroke_data_.append(data)
                original_ascii_data_.append(ascii)



        whole_data=list(zip(original_stroke_data_,original_ascii_data_))
        np.random.shuffle(whole_data)
        np.random.shuffle(whole_data)

        original_stroke_data,original_ascii_data=zip(*whole_data)

        average_ascii_sum=0
        average_tsteps_sum=0
        average_char_sum=0
        for i in range(len(original_stroke_data)):
            average_ascii_sum+=len(original_stroke_data[i])/len(original_ascii_data[i])
            average_tsteps_sum+=len(original_stroke_data[i])
            average_char_sum+=len(original_ascii_data[i])


        average_ascii=average_ascii_sum/len(original_stroke_data)
        average_tsteps=average_tsteps_sum/len(original_stroke_data)
        average_char=average_char_sum/len(original_stroke_data)

        for i in range(len(original_stroke_data)):
            data=original_stroke_data[i]
            ascii=original_ascii_data[i]

            cur_data_counter = cur_data_counter + 1
            # data = np.array(data, dtype=np.float32)
            if cur_data_counter % 20 == 0:
                self.valid_stroke_data.append(data)
                self.valid_ascii_data.append(ascii)
            else:
                self.stroke_data.append(data)
                self.ascii_data.append(ascii)

        # minus 1, since we want the ydata to be a shifted version of x data
        assert (len(original_ascii_data) == len(original_stroke_data)), "There should be a 1:1 correspondence between stroke data and ascii labels."
        self.num_batches = int(len(self.stroke_data) / self.batch_size)
        self.logger.write("\tload dataset:")
        self.logger.write("\t\t{} average number of characters per phrase".format(average_char))
        self.logger.write("\t\t{} average steps per phrase".format(average_tsteps))
        self.logger.write("\t\t{} average steps per ascii".format(average_ascii))
        self.logger.write("\t\t{} zero individual data points".format(zero))
        self.logger.write("\t\t{} train individual data points".format(len(self.stroke_data)))
        self.logger.write("\t\t{} valid individual data points".format(len(self.valid_stroke_data)))
        self.logger.write("\t\t{} batches".format(self.num_batches))


    def validation_data(self):
        # returns validation data
        x_batch = []
        y_batch = []
        ascii_list = []
        for i in range(self.batch_size):
            valid_ix = i%len(self.valid_stroke_data)
            data = self.valid_stroke_data[valid_ix]
            x_batch.append(np.copy(data[-self.tsteps-4:-4]))
            y_batch.append(np.copy(data[-self.tsteps-3:-3]))
            ascii_list.append(self.valid_ascii_data[valid_ix])
        one_hots = [to_one_hot(s, self.ascii_steps, self.alphabet) for s in ascii_list]
        return x_batch, y_batch, ascii_list, one_hots

    def next_batch(self):
        # returns a randomized, tsteps-sized portion of the training data
        x_batch = []
        y_batch = []
        ascii_list = []
        for i in range(self.batch_size):
            data = self.stroke_data[self.idx_perm[self.pointer]]
            #idx = random.randint(0, len(data)-self.tsteps-2)
            x_batch.append(np.copy(data[-self.tsteps-4:-4]))
            y_batch.append(np.copy(data[-self.tsteps-3:-3]))
            ascii_list.append(self.ascii_data[self.idx_perm[self.pointer]])
            self.tick_batch_pointer()
        one_hots = [to_one_hot(s, self.ascii_steps, self.alphabet) for s in ascii_list]
        return x_batch, y_batch, ascii_list, one_hots

    def tick_batch_pointer(self):
        self.pointer += 1
        if (self.pointer >= len(self.stroke_data)):
            self.reset_batch_pointer()
    def reset_batch_pointer(self):
        self.idx_perm = np.random.permutation(len(self.stroke_data))
        self.pointer = 0

def to_one_hot(s, ascii_steps, alphabet):
    alphabet_bi_list=[]
    for i in range(len(alphabet)):
        for j in range(len(alphabet)):
            alphabet_bi=alphabet[i]+alphabet[j]
            alphabet_bi_list.append(alphabet_bi)

    s_list=[]

    for i in range(len(s)):
        if i > len(s)-2:
            continue
        s_bi=s[i]+s[i+1]
        s_list.append(s_bi)


    seq = [alphabet_bi_list.index(char) + 1 for char in s_list]
    if len(seq) >= ascii_steps-1:
        # print(ascii_steps)
        seq = seq[-int(ascii_steps-1):]
        ss = [alphabet_bi_list[i - 1] for i in seq]
        ss=''.join(ss)

    else:
        seq = [0]*int(ascii_steps-1 - int(len(seq))) + seq
        ss=s

    one_hot = np.zeros((int(ascii_steps-1),len(alphabet_bi_list)+1))
    one_hot[np.arange(int(ascii_steps-1)),seq] = 1
    return one_hot


def to_one_hot_sample(s,tsteps, alphabet):
    ascii_steps=int(tsteps/12)
    alphabet_bi_list=[]
    for i in range(len(alphabet)):
        for j in range(len(alphabet)):
            alphabet_bi=alphabet[i]+alphabet[j]
            alphabet_bi_list.append(alphabet_bi)

    s_list=[]

    for i in range(len(s)):
        if i > len(s)-2:
            continue
        s_bi=s[i]+s[i+1]
        s_list.append(s_bi)


    seq = [alphabet_bi_list.index(char) + 1 for char in s_list]
    if len(seq) >= ascii_steps-1:
        # print(ascii_steps)
        seq = seq[-int(ascii_steps-1):]

    else:
        seq = [0]*int(ascii_steps-1 - int(len(seq))) + seq


    one_hot = np.zeros((int(ascii_steps-1),len(alphabet_bi_list)+1))
    one_hot[np.arange(int(ascii_steps-1)),seq] = 1
    return one_hot





def to_string_sample(s, tsteps, alphabet):
    # steplimit=3e3; s = s[:3e3] if len(s) > 3e3 else s # clip super-long strings
    ascii_steps = int(tsteps/12)
    seq = [alphabet.find(char) + 1 for char in s]
    if len(seq) >= ascii_steps:
        # print(ascii_steps)
        seq = seq[-int(ascii_steps):]
        ss = [alphabet[i - 1] for i in seq]
        ss=''.join(ss)

    else:
        seq = [0]*int(ascii_steps - int(len(seq))) + seq
        ss=s


    return ss






def to_string(s, ascii_steps, alphabet):
    seq = [alphabet.find(char) + 1 for char in s]
    if len(seq) >= ascii_steps:
        # print(ascii_steps)
        seq = seq[-int(ascii_steps):]
        ss = [alphabet[i - 1] for i in seq]
        ss=''.join(ss)

    else:
        seq = [0]*int(ascii_steps - int(len(seq))) + seq
        ss=s


    return ss


# abstraction for logging
class Logger():
    def __init__(self, args):
        self.logf = '{}train_scribe.txt'.format(args.log_dir) if args.train else '{}sample_scribe.txt'.format(args.log_dir)
        with open(self.logf, 'w') as f: f.write("project by shawn shen")

    def write(self, s, print_it=True):
        if print_it:
            print(s)
        with open(self.logf, 'a') as f:
            f.write(s + "\n")