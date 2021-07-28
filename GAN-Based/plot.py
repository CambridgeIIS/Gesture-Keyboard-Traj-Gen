import numpy as np
from preprocess import get_stroke_sequence
import matplotlib.pyplot as plt



def plot():
    labels = 'do_you_have_to_be_there'
    # true_data = get_stroke_sequence('/home/shawn/desktop/GAN_DE/holo_real/h_{}.csv'.format(labels),labels)
    true_data = get_stroke_sequence('/home/shawn/desktop/GAN_DE/processed_real_all/r_{}.csv'.format(labels), labels)
    fake_data = np.load('/home/shawn/desktop/GAN_DE/20210213-150805/res/save/g_{}_107.npy'.format(labels))
    # true_data = np.load('/home/shawn/desktop/GAN_DE/20210213-150805/res/save/f_{}.npy'.format(labels))

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title(labels)
    plt.plot(true_data[:, 0], true_data[:, 1], 'b-', linewidth=2.0)
    plt.subplot(1, 2, 2)
    plt.title(labels)
    plt.plot(fake_data[:, 0], fake_data[:, 1], 'b-', linewidth=2.0)
    plt.show()



if __name__ == "__main__":
    plot()