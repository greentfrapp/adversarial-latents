"""
- Build autoencoder
- Train autoencoder and visualize reconstruction
- Build classifier
- Train classifier
- Backprop adversarial gradient to latent space and generate reconstruction
"""

import tensorflow as tf
import numpy as np
from absl import app
from absl import flags
from keras.datasets import mnist
import PIL

FLAGS = flags.FLAGS

flags.DEFINE_bool('train_ae', False, 'Train autoencoder')
flags.DEFINE_bool('train_class', False, 'Train classifier')
flags.DEFINE_bool('reconstruct', False, 'Reconstruct with autoencoder')
flags.DEFINE_bool('test_class', False, 'Test classifier')
flags.DEFINE_bool('fgsm', False, 'Create adversarial sample')

class BaseModel():

	def save(self, sess, savepath, global_step=None, prefix='ckpt', verbose=False):
		if savepath[-1] != '/':
			savepath += '/'
		self.saver.save(sess, savepath + prefix, global_step=global_step)
		if verbose:
			print('Model saved to {}.'.format(savepath + prefix + '-' + str(global_step)))

	def load(self, sess, savepath, verbose=False):
		if savepath[-1] != '/':
			savepath += '/'
		ckpt = tf.train.latest_checkpoint(savepath)
		self.saver.restore(sess, ckpt)
		if verbose:
			print('Model loaded from {}.'.format(ckpt))

class Encoder(BaseModel):

	def __init__(self, inputs):
		self.name = 'encoder'
		self.inputs = inputs
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			self.build_model()
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
			self.saver = tf.train.Saver(var_list=variables, max_to_keep=1)

	def build_model(self):
		
		encoder_dense_1 = tf.layers.dense(
			inputs=self.inputs,
			units=1000,
			activation=tf.nn.relu,
			name='encoder_dense_1',
		)
		encoder_dense_2 = tf.layers.dense(
			inputs=encoder_dense_1,
			units=1000,
			activation=tf.nn.relu,
			name='encoder_dense_2',
		)

		self.outputs = tf.layers.dense(
			inputs=encoder_dense_2,
			units=64,
			activation=None,
			name='encoder_output',
		)

class Decoder(BaseModel):

	def __init__(self, inputs):
		self.name = 'decoder'
		self.inputs = inputs
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			self.build_model()
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
			self.saver = tf.train.Saver(var_list=variables, max_to_keep=1)

	def build_model(self):
		
		decoder_dense_1 = tf.layers.dense(
			inputs=self.inputs,
			units=1000,
			activation=tf.nn.relu,
			name='decoder_dense_1',
		)
		decoder_dense_2 = tf.layers.dense(
			inputs=decoder_dense_1,
			units=1000,
			activation=tf.nn.relu,
			name='decoder_dense_2',
		)

		self.outputs = tf.layers.dense(
			inputs=decoder_dense_2,
			units=784,
			activation=tf.sigmoid,
			name='decoder_output',
		)

class Classifier(BaseModel):

	def __init__(self, inputs):
		self.name = 'classifier'
		self.inputs = inputs
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			self.build_model()
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
			self.saver = tf.train.Saver(var_list=variables, max_to_keep=1)
	
	def build_model(self):

		classifier_dense_1 = tf.layers.dense(
			inputs=self.inputs,
			units=1000,
			activation=tf.nn.relu,
			name='classifier_dense_1',
		)
		classifier_dense_2 = tf.layers.dense(
			inputs=classifier_dense_1,
			units=1000,
			activation=tf.nn.relu,
			name='classifier_dense_2',
		)
		self.outputs = tf.layers.dense(
			inputs=classifier_dense_2,
			units=10,
			activation=None,
			name='classifier_output',
		)

def train_autoencoder():

	inputs = tf.placeholder(
		shape=(None, 28, 28),
		dtype=tf.float32,
		name='inputs',
	)

	encoder = Encoder(tf.reshape(inputs, [-1, 784]))
	latent = encoder.outputs
	decoder = Decoder(latent)
	reconstruction = tf.reshape(decoder.outputs, [-1, 28, 28])

	loss = tf.losses.mean_squared_error(labels=inputs, predictions=reconstruction)
	optimizer = tf.train.AdamOptimizer().minimize(loss)

	(x_train, _), (x_test, _) = mnist.load_data()

	n_steps = 10000
	batchsize = 128
	end = 0
	shuffle_x = np.random.RandomState(1)
	min_val_loss = np.inf

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	for step in np.arange(n_steps):
		start = end
		end = start + 128
		if end > len(x_train):
			start = 0
			end = 128
			shuffle_x.shuffle(x_train)
		minibatch_x = x_train[start:end]

		training_loss, _ = sess.run([loss, optimizer], {inputs: minibatch_x / 256.})

		if step > 0 and step % 100 == 0:
			print('Step #{} - Loss {:.3f}'.format(step, training_loss))

		if step > 0 and step % 500 == 0:
			minibatch_val_x = x_test[:128]
			shuffle_x.shuffle(x_test)
			val_loss = sess.run(loss, {inputs: minibatch_val_x / 256.})
			print('Step #{} - Validation Loss {:.3f}'.format(step, val_loss))
			if val_loss < min_val_loss:
				encoder.save(sess, 'saved_models/encoder/', global_step=step, verbose=True)
				decoder.save(sess, 'saved_models/decoder/', global_step=step, verbose=True)
				min_val_loss = val_loss

def reconstruct():

	inputs = tf.placeholder(
		shape=(None, 28, 28),
		dtype=tf.float32,
		name='inputs',
	)

	encoder = Encoder(tf.reshape(inputs, [-1, 784]))
	latent = encoder.outputs
	decoder = Decoder(latent)
	reconstruction = tf.reshape(decoder.outputs, [-1, 28, 28])

	sess = tf.Session()
	encoder.load(sess, 'saved_models/encoder/', verbose=True)
	decoder.load(sess, 'saved_models/decoder/', verbose=True)

	(x_train, _), (x_test, _) = mnist.load_data()

	sample = np.expand_dims(x_test[np.random.choice(len(x_test))], axis=0)

	generated_sample = sess.run(reconstruction, {inputs: sample / 256.}) * 256.

	image = np.concatenate([sample[0], generated_sample[0]], axis=1)

	PIL.Image.fromarray(image).resize((200, 100)).show()

def train_classifier():

	inputs = tf.placeholder(
		shape=(None, 28, 28),
		dtype=tf.float32,
		name='inputs',
	)
	labels = tf.placeholder(
		shape=(None),
		dtype=tf.int64,
		name='labels',
	)

	classifier = Classifier(tf.reshape(inputs, [-1, 784]))
	logits = classifier.outputs

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(labels, depth=10), logits=logits))
	optimizer = tf.train.AdamOptimizer().minimize(loss)

	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	n_steps = 10000
	batchsize = 128
	end = 0
	shuffle_x = np.random.RandomState(1)
	shuffle_y = np.random.RandomState(1)
	min_val_loss = np.inf

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	for step in np.arange(n_steps):
		start = end
		end = start + 128
		if end > len(x_train):
			start = 0
			end = 128
			shuffle_x.shuffle(x_train)
			shuffle_y.shuffle(y_train)
		minibatch_x = x_train[start:end]
		minibatch_y = y_train[start:end]

		training_loss, _ = sess.run([loss, optimizer], {inputs: minibatch_x / 256., labels: minibatch_y})

		if step > 0 and step % 100 == 0:
			print('Step #{} - Loss {:.3f}'.format(step, training_loss))

		if step > 0 and step % 500 == 0:
			minibatch_val_x = x_test[:128]
			minibatch_val_y = y_test[:128]
			shuffle_x.shuffle(x_test)
			shuffle_y.shuffle(y_test)
			val_loss = sess.run(loss, {inputs: minibatch_val_x / 256., labels: minibatch_val_y})
			print('Step #{} - Validation Loss {:.3f}'.format(step, val_loss))
			if val_loss < min_val_loss:
				classifier.save(sess, 'saved_models/classifier/', global_step=step, verbose=True)
				min_val_loss = val_loss

def test_classifier():

	inputs = tf.placeholder(
		shape=(None, 28, 28),
		dtype=tf.float32,
		name='inputs',
	)
	labels = tf.placeholder(
		shape=(None),
		dtype=tf.int64,
		name='labels',
	)

	classifier = Classifier(tf.reshape(inputs, [-1, 784]))
	logits = classifier.outputs

	accuracy = tf.contrib.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, axis=1))

	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	n_steps = 10000
	batchsize = 128
	end = 0

	sess = tf.Session()
	classifier.load(sess, 'saved_models/classifier/', verbose=True)

	overall_acc = []
	for step in np.arange(n_steps):
		start = end
		end = start + 128
		if end > len(x_train):
			break
		minibatch_x = x_train[start:end]
		minibatch_y = y_train[start:end]

		test_acc = sess.run(accuracy, {inputs: minibatch_x / 256., labels: minibatch_y})
		overall_acc.append(test_acc)

		if step > 0 and step % 100 == 0:
			print('Step #{} - Acc {:.3f}'.format(step, test_acc))

	print('Overall Accuracy: {:.3f}'.format(np.mean(overall_acc)))

def latent_fgsm():
	inputs = tf.placeholder(
		shape=(None, 28, 28),
		dtype=tf.float32,
		name='inputs',
	)
	labels = tf.placeholder(
		shape=(None),
		dtype=tf.int64,
		name='labels',
	)
	latent = tf.placeholder(
		shape=(None, 64),
		dtype=tf.float32,
		name='latent',
	)
	adv_latent = tf.placeholder(
		shape=(None, 64),
		dtype=tf.float32,
		name='adv_latent',
	)

	encoder = Encoder(tf.reshape(inputs, [-1, 784]))
	encoding = encoder.outputs

	decoder = Decoder(latent)
	reconstruction = tf.reshape(decoder.outputs, [-1, 28, 28])
	classifier = Classifier(tf.reshape(reconstruction, [-1, 784]))
	logits = classifier.outputs
	softmax = tf.nn.softmax(logits, axis=1)
	prediction = tf.argmax(logits, axis=1)

	fgsm_classifier = Classifier(tf.reshape(inputs, [-1, 784]))
	fgsm_logits = fgsm_classifier.outputs
	fgsm_softmax = tf.nn.softmax(fgsm_logits, axis=1)
	fgsm_prediction = tf.argmax(fgsm_logits, axis=1)

	classifier_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(labels, depth=10), logits=logits))
	latent_gradient = tf.gradients(classifier_loss, latent)

	fgsm_classifier_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(labels, depth=10), logits=fgsm_logits))
	fgsm_gradient = tf.gradients(fgsm_classifier_loss, inputs)

	sess = tf.Session()
	encoder.load(sess, 'saved_models/encoder/', verbose=True)
	decoder.load(sess, 'saved_models/decoder/', verbose=True)
	classifier.load(sess, 'saved_models/classifier/', verbose=True)
	fgsm_classifier.load(sess, 'saved_models/classifier/', verbose=True)

	_, (x_test, y_test) = mnist.load_data()

	sample = np.expand_dims(x_test[0], axis=0)
	label = np.expand_dims(y_test[0], axis=0)
	fake_label = np.array([3])

	original_encoding = sess.run(encoding, {inputs: sample / 256.})
	new_encoding = original_encoding

	for i in np.arange(500):
		dLdz = sess.run(latent_gradient, {latent: new_encoding, labels: fake_label})
		new_encoding = new_encoding - 1e-2 * dLdz[0]

	old_prediction = sess.run(prediction, {latent: original_encoding})

	new_prediction, new_softmax, new_image = sess.run([prediction, softmax, reconstruction], {latent: new_encoding})

	print(old_prediction)
	print(new_prediction)
	print(new_softmax)

	delta = new_image.reshape(28, 28) * 256. - sample.reshape(28, 28) + 128.
	new_image = new_image.reshape(28, 28) * 256.
	old_image = sample.reshape(28, 28)

	images = np.concatenate([old_image, delta, new_image], axis=1)

	PIL.Image.fromarray(images).resize((300, 100)).convert('RGBA').save('latent_adversarial.png')

	# FGSM

	new_sample = sample / 256.
	for i in np.arange(100):
		dIdz = sess.run(fgsm_gradient, {inputs: new_sample, labels: fake_label})
		new_sample = new_sample - 1e-3 * np.sign(dIdz[0])

	new_fgsm_prediction, new_fgsm_softmax = sess.run([fgsm_prediction, fgsm_softmax], {inputs: new_sample})
	print(new_fgsm_prediction)
	print(new_fgsm_softmax)

	delta = new_sample.reshape(28, 28) * 256. - sample.reshape(28, 28) + 128.
	new_image = new_sample.reshape(28, 28) * 256.
	old_image = sample.reshape(28, 28)

	images = np.concatenate([old_image, delta, new_image], axis=1)

	PIL.Image.fromarray(images).resize((300, 100)).convert('RGBA').save('reg_adversarial.png')

def main(unused_args):

	if FLAGS.train_ae:
		train_autoencoder()
	elif FLAGS.reconstruct:
		reconstruct()
	elif FLAGS.train_class:
		train_classifier()
	elif FLAGS.test_class:
		test_classifier()
	elif FLAGS.fgsm:
		latent_fgsm()

if __name__ == '__main__':
	app.run(main)
