import tensorflow as tf
import logging
import os

import time
from datetime import datetime
from inference import generate_images
import data_utils

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def train(buffer_size, batch_size, epochs, real_dataset,fake_dataset,
          generator_g, generator_f,
          discriminator_g, discriminator_f,
          discriminator_optimizer_f, discriminator_optimizer_g,
          model_path, checkpoint, checkpoint_prefix,train_log_dir,
         loss_fn, disc_iters, generator_optimizer_g, generator_optimizer_f,max_c_length,
          max_x_length, ckpt_path, restore, manager, gen_path):
    # Generator G translates X -> Y
    # Generator F translates Y -> X.


    batch_per_epoch = int(buffer_size / batch_size)+1

    logging.info('training.....')

    if not os.path.exists(checkpoint_prefix):
        os.makedirs(checkpoint_prefix)
        print('check point not exist {}'.format(checkpoint_prefix))
    else:
        print('check point {}'.format(checkpoint_prefix))

    if not os.path.exists(model_path):
        os.makedirs(model_path)
        print('model path not exist {}'.format(model_path))
    else:
        print('model path {}'.format(model_path))

    if restore:
        checkpoint.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")
        # status = checkpoint.restore(tf.train.latest_checkpoint(ckpt_path))
        # logging.info('restoring form {}'.format(tf.train.latest_checkpoint(ckpt_path)))


    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    for epoch_idx in range(epochs):
        start = time.time()

        for batch_idx in range(batch_per_epoch):
            real_input, real_labels = next(real_dataset)
            fake_input, fake_labels = next(fake_dataset)
            train_step(batch_size, real_input, real_labels,fake_input, fake_labels,  generator_f,generator_g, discriminator_f, discriminator_g,
                       loss_fn, epoch_idx, batch_idx, batch_per_epoch, discriminator_optimizer_f, discriminator_optimizer_g, disc_iters,
                       generator_optimizer_g, generator_optimizer_f,  max_c_length,
                       max_x_length,train_summary_writer, gen_path, checkpoint, manager)


        #
        # checkpoint.step.assign_add(1)
        #
        #
        # # Save the model every 5 epochs
        # if (epoch_idx + 1) % 1 == 0:
        #     save_path = manager.save()
        #     logging.info("Saved checkpoint for epoch {}: {}".format(int(checkpoint.step), save_path))
        #     # checkpoint.save(file_prefix=checkpoint_prefix)
        #     # logging.info('model saved at check path {}'.format(ckpt_path))



        logging.info('Time for epoch {} is {} sec'.format(epoch_idx + 1, time.time() - start))

        # save generator model
        saved_path_g = model_path + 'generator_g_{}'.format(epoch_idx)
        saved_path_f = model_path + 'generator_f_{}'.format(epoch_idx)
        # tf.saved_model.save(generator,saved_path )
        generator_g.save(saved_path_g)
        generator_f.save(saved_path_f)


def calc_cycle_loss(real_input, cycled_input):
    LAMBDA = 10
    loss1 = tf.reduce_mean(tf.abs(real_input - cycled_input))

    return LAMBDA * loss1


def identity_loss(real_input, same_imput):
    LAMBDA = 10
    loss = tf.reduce_mean(tf.abs(real_input - same_imput))
    return LAMBDA * 0.5 * loss

def discriminator_loss(real, generated):
  real_loss = loss_obj(tf.ones_like(real), real)

  generated_loss = loss_obj(tf.zeros_like(generated), generated)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss * 0.5

def generator_loss(generated):
  return loss_obj(tf.ones_like(generated), generated)



# @tf.function
def train_step(batch_size, real_input, real_labels,fake_input, fake_labels,
               generator_f, generator_g, discriminator_f, discriminator_g,
                       loss_fn, epoch_idx, batch_idx, batch_per_epoch,
               discriminator_optimizer_f, discriminator_optimizer_g, disc_iters,
                       generator_optimizer_g, generator_optimizer_f,  max_c_length,
                       max_x_length,train_summary_writer, gen_path, checkpoint, manager):



    start = time.time()

    real_x = fake_input
    x_labels = fake_labels



    real_y = real_input
    y_labels = real_labels

    with tf.GradientTape(persistent=True) as tape:
        fake_x = generator_g([real_x, x_labels], training = True)
        cycled_x = generator_f([fake_x, x_labels], training = True)

        fake_y = generator_f([real_y, y_labels], training = True)
        cycled_y = generator_g([fake_y, y_labels], training = True)

        same_x = generator_f([real_x, x_labels], training = True)
        same_y = generator_g([real_y, y_labels], training = True)

        disc_real_x = discriminator_g([real_x, x_labels], training = True)
        disc_real_y = discriminator_f([real_y, y_labels], training = True)

        disc_fake_y = discriminator_g([fake_y, y_labels], training=True)
        disc_fake_x = discriminator_f([fake_x, x_labels], training=True)

        gen_g_loss = generator_loss(disc_fake_x)
        gen_f_loss = generator_loss(disc_fake_y)

        total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

        # Total generator loss = adversarial loss + cycle loss
        # total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y) + identity_loss(real_x, fake_x)
        # total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x) + identity_loss(real_y, fake_y)

        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_y)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_x)

        disc_x_loss_mean = tf.reduce_mean(disc_x_loss)
        disc_y_loss_mean = tf.reduce_mean(disc_y_loss)
        total_gen_g_loss_mean = tf.reduce_mean(total_gen_g_loss)
        total_gen_f_loss_mean = tf.reduce_mean(total_gen_f_loss)

    global_step = batch_idx + 1 + (epoch_idx * batch_per_epoch)

    # checkpoint.step.assign_add(1)

    if (global_step + 1) % 10 == 0:
        logging.info('saving generated images to {}'.format(gen_path))
        fake_str = data_utils.decode_ascii(x_labels)
        real_str = data_utils.decode_ascii(y_labels)
        for i in range(5):
            generate_images(real_x[i], fake_x[i], fake_str[i], gen_path , global_step,'g')
            generate_images(real_y[i], fake_y[i], real_str[i], gen_path, global_step,'f')

        save_path = manager.save()
        logging.info("Saved checkpoint for step {}: {}".format(int(checkpoint.step), save_path))

        step_time = time.time() - start
        logging.info(
            '%.3f>%d, %d/%d,  disc_x_loss=%.4f, disc_y_loss=%.4f, total_gen_g_loss=%.4f, total_gen_g_loss=%.4f' % (
            step_time,
            epoch_idx + 1, batch_idx + 1, batch_per_epoch, disc_x_loss_mean.numpy(), disc_y_loss_mean.numpy(),
            total_gen_g_loss_mean.numpy(), total_gen_f_loss_mean.numpy()))



    if (global_step + 1) % 10 == 0:
        with train_summary_writer.as_default():
            tf.summary.scalar('disc_x_loss_mean', disc_x_loss_mean, step=global_step)
            tf.summary.scalar('disc_y_loss_mean', disc_y_loss_mean, step=global_step)
            tf.summary.scalar('total_gen_g_loss_mean', total_gen_g_loss_mean, step=global_step)
            tf.summary.scalar('total_gen_f_loss_mean', total_gen_f_loss_mean, step=global_step)



    # Calculate the gradients for generator and discriminator
    generator_g_gradients = tape.gradient(total_gen_g_loss,
                                          generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss,
                                          generator_f.trainable_variables)

    discriminator_f_gradients = tape.gradient(disc_y_loss,
                                              discriminator_f.trainable_variables)
    discriminator_g_gradients = tape.gradient(disc_x_loss,
                                              discriminator_g.trainable_variables)

    # Apply the gradients to the optimizer
    generator_optimizer_f.apply_gradients(zip(generator_g_gradients,
                                              generator_g.trainable_variables))

    generator_optimizer_g.apply_gradients(zip(generator_f_gradients,
                                              generator_f.trainable_variables))

    discriminator_optimizer_f.apply_gradients(zip(discriminator_f_gradients,
                                                  discriminator_f.trainable_variables))

    discriminator_optimizer_g.apply_gradients(zip(discriminator_g_gradients,
                                                  discriminator_g.trainable_variables))




