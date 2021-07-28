
from cycle_train import train

from data_utils import load_str, convert2labels, fake_generator
from inference import generate_images

import tensorflow as tf
from data_utils import load_prepare_data_real, load_prepare_data_fake
import os
from net_architecture import make_generator_no_label , make_discriminator_no_label
import gin
from datetime import datetime
import logging

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#
# tf.config.set_soft_device_placement(True)
# tf.debugging.set_log_device_placement(True)

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[1], True)
tf.config.run_functions_eagerly(False)


from net_loss import hinge, not_saturating
gin.external_configurable(hinge)
gin.external_configurable(not_saturating)

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

@gin.configurable
def setup_optimizer(g_lr, d_lr, beta_1, beta_2, loss_fn, disc_iters):
    generator_optimizer_f = tf.keras.optimizers.Adam(learning_rate=g_lr, beta_1=beta_1, beta_2=beta_2)
    generator_optimizer_g = tf.keras.optimizers.Adam(learning_rate=g_lr, beta_1=beta_1, beta_2=beta_2)
    discriminator_optimizer_f = tf.keras.optimizers.Adam(learning_rate=d_lr, beta_1=beta_1, beta_2=beta_2)
    discriminator_optimizer_g = tf.keras.optimizers.Adam(learning_rate=d_lr, beta_1=beta_1, beta_2=beta_2)
    return generator_optimizer_f, generator_optimizer_g, discriminator_optimizer_f, discriminator_optimizer_g,\
           loss_fn, disc_iters



@gin.configurable('shared_specs')
def get_shared_specs(batch_sz, enc_units, param_dim, dec_units, max_c_length, max_x_length, epochs, input_dim):

    return batch_sz, enc_units, param_dim, dec_units, max_c_length, max_x_length, epochs, input_dim


@gin.configurable('io')
def setup_io(base_path, checkpoint_dir, gen_imgs_dir, model_dir,  buffer_size, raw_dir, read_dir,
             log_dir, train_dir, restore):
    if restore:
        date_dir = '20210615-184146'
    else:
        date_dir = datetime.now().strftime("%Y%m%d-%H%M%S")

    gen_path = base_path + date_dir + gen_imgs_dir
    ckpt_path = base_path + date_dir + checkpoint_dir
    model_path = base_path + date_dir + model_dir
    train_log_dir = base_path + date_dir + train_dir
    raw_dir = base_path + date_dir + raw_dir
    read_dir = base_path + date_dir + read_dir
    log_dir = base_path + date_dir + log_dir
    return  ckpt_path, gen_path, model_path, raw_dir, read_dir, \
            log_dir,train_log_dir, buffer_size, restore


def init_logging(log_dir):
    logging_level = logging.INFO

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    date_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
    log_file = 'log_{}.txt'.format(date_str)

    logging.basicConfig(
        filename=os.path.join(log_dir, log_file),
        level = logging_level,
        format='[[%(asctime)s]] %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p'
    )
    logging.getLogger().addHandler(logging.StreamHandler())



def main():
    # init params
    gin.parse_config_file('de_gan.gin')
    batch_sz, enc_units, param_dim, dec_units, max_c_length, max_x_length, epochs, input_dim  = get_shared_specs()
    ckpt_path, gen_path, model_path, raw_dir, read_dir, log_dir, train_log_dir, buffer_size, restore = setup_io()



    init_logging(log_dir)

    logging.info('batch_sz : %d , enc_units: %d, param_dim %d , dec_units %d , max_c_length %d , max_x_length %d , epochs %d , input_dim %d '%(
        batch_sz, enc_units, param_dim, dec_units, max_c_length, max_x_length, epochs, input_dim))
    # load and preprocess dataset (python generator)
    real_dataset = load_prepare_data_real(batch_sz, max_x_length, max_c_length)
    fake_dataset = load_prepare_data_real(batch_sz, max_x_length, max_c_length)

    # init generator, discriminator and recognizer
    generator_g = make_generator_no_label(enc_units, batch_sz, param_dim, dec_units, gen_path, max_x_length, max_c_length,
                                   input_dim, vis_model = True)
    generator_f = make_generator_no_label(enc_units, batch_sz, param_dim, dec_units, gen_path, max_x_length, max_c_length,
                                   input_dim, vis_model=True)


    discriminator_g = make_discriminator_no_label(enc_units, batch_sz, param_dim, dec_units, gen_path, max_x_length,
                                         max_c_length, vis_model = True)
    discriminator_f = make_discriminator_no_label(enc_units, batch_sz, param_dim, dec_units, gen_path, max_x_length,
                                         max_c_length, vis_model=True)



    # init optimizer for both generator, discriminator and recognizer
    generator_optimizer_f, generator_optimizer_g, discriminator_optimizer_f, discriminator_optimizer_g, \
    loss_fn, disc_iters = setup_optimizer()

    # purpose: save and restore models
    checkpoint_prefix = os.path.join(ckpt_path, "ckpt")

    checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
                                     generator_optimizer_f=generator_optimizer_f,
                                     generator_optimizer_g = generator_optimizer_g,
                                     discriminator_optimizer_f = discriminator_optimizer_f,
                                     discriminator_optimizer_g = discriminator_optimizer_g,
                                     generator_g = generator_g,
                                     generator_f = generator_f,
                                     discriminator_g = discriminator_g,
                                     discriminator_f = discriminator_f
                                     )

    manager = tf.train.CheckpointManager(checkpoint, ckpt_path, max_to_keep=50)

    # start training
    # start training
    training = True
    if training == True:
        print('training mode')
        train(buffer_size, batch_sz, epochs, real_dataset, fake_dataset,
              generator_g, generator_f,
              discriminator_g, discriminator_f,
              discriminator_optimizer_f, discriminator_optimizer_g,
              model_path, checkpoint, checkpoint_prefix,train_log_dir,
             loss_fn, disc_iters, generator_optimizer_g, generator_optimizer_f,max_c_length,
              max_x_length, ckpt_path, restore, manager, gen_path)
    else:
        print('infernece mode')
        checkpoint.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))

        # inference loop

        fake_str = load_str(batch_sz)
        fake_labels = convert2labels(fake_str, max_c_length, batch_sz)
        fake_input = fake_generator(fake_str, max_x_length, batch_sz)

        # run inference process
        predictions = generator_f([fake_input, fake_labels], training=False)

        # plot results
        for i in range(batch_sz):
            generate_images(fake_input[i], predictions[i], fake_str[i], gen_path)
            # print(fake_input[i])




if __name__ == "__main__":
    main()
