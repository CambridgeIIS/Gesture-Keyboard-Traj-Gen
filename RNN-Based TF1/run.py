import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import argparse
import time
import os
from datetime import datetime
from model import Model
from utils import *
from sample import *

def main():
	parser = argparse.ArgumentParser()
	TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

	new = False


	if new == True:
		os.mkdir(TIMESTAMP)
		os.chdir(TIMESTAMP)
		os.mkdir('data/')
		os.makedirs('logs/figures/')
		os.makedirs('logs/stroke/')
		os.mkdir('saved/')
	else:
		TIMESTAMP = '2020-09-07T11-52-46'
		os.chdir(TIMESTAMP)




	#general model params
	parser.add_argument('--train', dest='train', action='store_true', help='train the model')
	parser.add_argument('--sample', dest='train', action='store_false', help='sample from the model')
	parser.add_argument('--rnn_size', type=int, default=400, help='size of RNN hidden state')
	parser.add_argument('--tsteps', type=int, default=240, help='RNN time steps (for backprop)')
	parser.add_argument('--nmixtures', type=int, default=8, help='number of gaussian mixtures')
	parser.add_argument('--lamda', type=float, default=0.01, help='regularization constant')
	parser.add_argument('--threshold', type=float, default=5.0, help='threshold for the offset')
	parser.add_argument('--scale', type=float, default=100.0, help='scale the data to make offset range between -1 to 1')
	# window params
	parser.add_argument('--kmixtures', type=int, default=10, help='number of gaussian mixtures for character window')
	parser.add_argument('--alphabet', type=str, default=' abcdefghijklmnopqrstuvwxyz_', \
						help='default is a-z, A-Z, space, and <UNK> tag')
	parser.add_argument('--tsteps_per_ascii', type=int, default=12, help='expected number of pen points per character')

	# training params

	parser.add_argument('--batch_size', type=int, default=32, help='batch size for each gradient step')
	parser.add_argument('--nbatches', type=int, default=1000, help='number of batches per epoch')
	parser.add_argument('--nepochs', type=int, default=250, help='number of epochs')
	parser.add_argument('--dropout', type=float, default=0.8, help='probability of keeping neuron during dropout')

	parser.add_argument('--grad_clip', type=float, default=10., help='clip gradients to this magnitude')
	parser.add_argument('--optimizer', type=str, default='rmsprop', help="ctype of optimizer: 'rmsprop' 'adam'")
	parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate')
	parser.add_argument('--lr_decay', type=float, default=0.9, help='decay rate for learning rate')
	parser.add_argument('--decay', type=float, default=0.95, help='decay rate for rmsprop')
	parser.add_argument('--momentum', type=float, default=0.9, help='momentum for rmsprop')

	#book-keeping
	parser.add_argument('--data_scale', type=int, default=50, help='amount to scale data down before training')
	parser.add_argument('--log_dir', type=str, default='./logs/stroke', help='location, relative to execution, of log files')
	parser.add_argument('--data_dir', type=str, default='./data', help='location, relative to execution, of data')
	parser.add_argument('--save_path', type=str, default='saved/model.ckpt', help='location to save model')
	parser.add_argument('--save_every', type=int, default=500, help='number of batches between each save')

	#sampling
	parser.add_argument('--text', type=str, default='', help='string for sampling model (defaults to test cases)', nargs='+')
	parser.add_argument('--style', type=int, default=71, help='optionally condition model on a preset style')
	parser.add_argument('--bias', type=float, default=1.0, help='higher bias means neater, lower means more diverse (range is 0-5)')
	parser.add_argument('--sleep_time', type=int, default=60*5, help='time to sleep between running sampler')
	parser.set_defaults(train=True)
	args = parser.parse_args()
        

	train_model(args) if args.train else sample_model(args)

def train_model(args):
	tf.reset_default_graph()
	logger = Logger(args) # make logging utility
	logger.write("\nTRAINING MODE...")
	logger.write("{}\n".format(args))
	logger.write("loading data...")
	data_loader = DataLoader(args, logger=logger)
	
	logger.write("building model...")
	model = Model(args, logger=logger)

	logger.write("attempt to load saved model...")
	load_was_success, global_step = model.try_load_model(args.save_path)

	v_x, v_y, v_s, v_c = data_loader.validation_data()
	valid_inputs = {model.input_data: v_x, model.target_data: v_y, model.char_seq: v_c}

	logger.write("training...")


	# TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
	#
	# train_log_dir = args.log_dir + TIMESTAMP
	#
	# model_log_dir = './saved/' + TIMESTAMP+'model.ckpt'
	train_log_dir = args.log_dir + 'train/'
	valid_log_dir = args.log_dir + 'valid/'

	summary_writer = tf.summary.FileWriter(train_log_dir)
	summary_writer.add_graph(model.sess.graph)

	summary_writer_loss = tf.summary.FileWriter(valid_log_dir)


	model.sess.run(tf.assign(model.decay, args.decay ))
	model.sess.run(tf.assign(model.momentum, args.momentum ))
	running_average = 0.0 ; remember_rate = 0.99
	for e in range(int(global_step/args.nbatches), args.nepochs):
		model.sess.run(tf.assign(model.learning_rate, args.learning_rate * (args.lr_decay ** e)))
		logger.write("learning rate: {}".format(model.learning_rate.eval()))

		c0, c1, c2 = model.istate_cell0.c.eval(), model.istate_cell1.c.eval(), model.istate_cell2.c.eval()
		h0, h1, h2 = model.istate_cell0.h.eval(), model.istate_cell1.h.eval(), model.istate_cell2.h.eval()
		kappa = np.zeros((args.batch_size, args.kmixtures, 1))

		for b in range(int(global_step%args.nbatches), args.nbatches):
			
			i = e * args.nbatches + b
			if global_step is not 0 : i+=1 ; global_step = 0

			if i % args.save_every == 0 and (i > 0):
				model.saver.save(model.sess, args.save_path, global_step = i) ; logger.write('SAVED MODEL')

			start = time.time()
			x, y, s, c = data_loader.next_batch()


			feed = {model.input_data: x, model.target_data: y, model.char_seq: c, model.init_kappa: kappa, \
					model.istate_cell0.c: c0, model.istate_cell1.c: c1, model.istate_cell2.c: c2, \
					model.istate_cell0.h: h0, model.istate_cell1.h: h1, model.istate_cell2.h: h2}

			[train_loss, _] = model.sess.run([model.cost, model.train_op], feed)
			feed.update(valid_inputs)
			feed[model.init_kappa] = np.zeros((args.batch_size, args.kmixtures, 1))
			[valid_loss] = model.sess.run([model.cost], feed)


			running_average = running_average*remember_rate + train_loss*(1-remember_rate)

			train_loss_ = tf.Summary()
			train_loss_.value.add(tag="loss", simple_value=train_loss)
			summary_writer.add_summary(train_loss_,i)
			valid_loss_ = tf.Summary()
			valid_loss_.value.add(tag="loss", simple_value=valid_loss)
			summary_writer_loss.add_summary(valid_loss_,i)

			summary = model.sess.run(model.merged)

			end = time.time()
			if i % 10 is 0: logger.write("{}/{}, loss = {:.3f}, regloss = {:.5f}, valid_loss = {:.3f}, time = {:.3f}" \
				.format(i, args.nepochs * args.nbatches, train_loss, running_average, valid_loss, end - start) )

def sample_model(args, logger=None):


	logger = Logger(args) if logger is None else logger # instantiate logger, if None
	logger.write("\nSAMPLING MODE...")
	logger.write("loading data...")
	
	logger.write("building model...")
	model = Model(args, logger)

	logger.write("attempt to load saved model...")
	load_was_success, global_step = model.try_load_model(args.save_path)

	
	if load_was_success:
		# for s in strings:
		s=' '.join(args.text)
		strokes, phis, windows, kappas,ss = sample(s, model, args)

		s_save_path = '{}/iter-{}-s-{}.txt'.format(args.log_dir, global_step, ss.replace(' ', '_'))


		f = open(s_save_path, "w")
		f.write("# x y\n")
		np.savetxt(f, np.array([strokes[:,0],strokes[:,1]]).T)
		f.close()
		print(s_save_path)

	else:
		logger.write("load failed, sampling canceled")
	quit()



if __name__ == '__main__':
	main()
