
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from data_utils import offsets_to_coords
from data_utils import load_prepare_data_real, decode_ascii, load_prepare_data_fake
import tensorflow as tf
import os
import numpy as np
import os
from preprocess import get_stroke_sequence


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def generate_images(fake_offsets, gen_strokes, labels,gen_path, global_step, generator_type):
    # fake_strokes = offsets_to_coords(fake_offsets)
    # gen_strokes = offsets_to_coords(gem_offsets)
    fake_strokes = fake_offsets
    gen_strokes = gen_strokes
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title(labels)
    plt.plot(fake_strokes[:, 0], fake_strokes[:, 1], 'b-', linewidth=2.0)
    plt.subplot(1, 2, 2)
    plt.title(labels)
    plt.plot(gen_strokes[:, 0], gen_strokes[:, 1], 'b-', linewidth=2.0)
    phrase_string = '_'.join(str(labels).split())
    plt.savefig(gen_path+'/{}_{}_{}.png'.format(generator_type, phrase_string, global_step))
    plt.close()


def save_data( gen_data, labels,gen_path, global_step, generator_type):
    # gen_strokes = offsets_to_coords(gen_offsets)
    phrase_string = '_'.join(str(labels).split())
    save_name = gen_path + '/{}_{}_{}.npy'.format(generator_type, phrase_string, global_step)
    print('saved to {}'.format(save_name))
    np.save(save_name, gen_data)


def main():
    batch_sz = 64
    max_x_length = 320
    max_c_length = 20

    time_date = '20210615-184146'
    path_to_saved_model_g = '/home/shawn/desktop/GAN_DE/{}/res/model/generator_g_1'.format(time_date)
    path_to_saved_model_f = '/home/shawn/desktop/GAN_DE/{}/res/model/generator_f_1'.format(time_date)

    save_path = '/home/shawn/desktop/GAN_DE/{}/res/save'.format(time_date)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # real_dataset = load_prepare_data_fake(batch_sz, max_x_length, max_c_length)
    real_dataset = load_prepare_data_real(batch_sz, max_x_length, max_c_length, 'real')

    imported_model_g = tf.saved_model.load(path_to_saved_model_g)
    imported_model_f = tf.saved_model.load(path_to_saved_model_f)

    # inference loop
    for j in range(1000):
        real_input, real_labels_ = next(real_dataset)
        real_input = np.array(real_input, np.float32)
        real_labels = np.array(real_labels_, np.float32)
        real_labels = np.reshape(real_labels,[batch_sz, max_c_length, 1])
        # run inference process
        predictions_g = imported_model_g([real_input, real_labels], training=False)
        predictions_f = imported_model_f([real_input, real_labels], training=False)


        labels_str = decode_ascii(real_labels_)

        for i in range(batch_sz):

            save_data(predictions_g[i], labels_str[i], save_path , int(i+j*batch_sz),'g')
            save_data(predictions_f[i], labels_str[i], save_path, int(i+j*batch_sz), 'f')
            generate_images(predictions_g[i], predictions_f[i], labels_str[i], save_path, int(i+j*batch_sz),'g')



if __name__ == "__main__":
    main()