import glob
import numpy as np
import drawing
import prepare_data

path = '../preprocess/preprocessed_data_all/*.csv'
for i, fname in enumerate(sorted(glob.glob(path), reverse=True)):
    if i%55==0:

        style=i

        phrases = fname.split('.')[2]
        phrases = phrases.split('_')[3:]
        phrases = ' '.join(phrases)
        print(phrases)
        print(style)
        x_p = prepare_data.get_stroke_sequence(fname)
        c_p = phrases.encode()

        np.save('styles/style-{}-strokes.npy'.format(style), x_p)
        np.save('styles/style-{}-chars.npy'.format(style), c_p)