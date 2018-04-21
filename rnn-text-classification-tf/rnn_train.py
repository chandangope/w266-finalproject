#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_rnn import TextRNN
from tensorflow.contrib import learn


# Parameters
# ==================================================

# Data loading params
flags = tf.app.flags
FLAGS = flags.FLAGS


# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 100)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 3, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


# Data Preparation
# ==================================================
# Load data
print("\nLoading train data...")
x_train, y_train = data_helpers.load_splitted_data_and_labels('../data/toxic_yes_train.txt', '../data/toxic_no_train.txt')
print("x_train length:{0}, y_train shape:{1}".format(len(x_train), y_train.shape))
print(x_train[0], y_train[0])

print("\nLoading dev data...")
x_dev, y_dev = data_helpers.load_splitted_data_and_labels('../data/toxic_yes_dev.txt', '../data/toxic_no_dev.txt')
print("x_dev length:{0}, y_dev shape:{1}".format(len(x_dev), y_dev.shape))
print(x_dev[-1], y_dev[-1])

x = x_train+x_dev
print("x length:{0}".format(len(x)))

# Build vocabulary
# max_sent_length, sent = max([(len(i.split(" ")),i) for i in x])
# print("Max sent length = {0}".format(max_sent_length))
# print("Sent with max length = {0}".format(sent))
max_sent_length = 40
vocab_processor = learn.preprocessing.VocabularyProcessor(max_sent_length)
x = np.array(list(vocab_processor.fit_transform(x))) #x is an iterable, [n_samples, max_sent_length] Word-id matrix.
print("Shape of word-id matrix: {0}".format(x.shape))

#Transform x_train and x_dev to word-id matrix
x_train = np.array(list(vocab_processor.transform(x_train)))
print("Shape of x_train matrix: {0}".format(x_train.shape))
x_dev = np.array(list(vocab_processor.transform(x_dev)))
print("Shape of x_dev matrix: {0}".format(x_dev.shape))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y_train)))
x_train = x_train[shuffle_indices]
y_train = y_train[shuffle_indices]

del x

vocabsize = len(vocab_processor.vocabulary_)
print("Vocabulary Size: {:d}".format(vocabsize))

# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        rnn = TextRNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=vocabsize,
            embedding_size=FLAGS.embedding_dim)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = optimizer.compute_gradients(rnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        # timestamp = str(int(time.time()))
        # out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs"))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", rnn.loss)
        acc_summary = tf.summary.scalar("accuracy", rnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        vocabulary = vocab_processor.vocabulary_
        initEmbeddings = data_helpers.load_embedding_vectors_glove(vocabulary)
        sess.run(rnn.W_embed.assign(initEmbeddings))

        for v in tf.trainable_variables():
            print(v.name)

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              rnn.input_x: x_batch,
              rnn.input_y: y_batch,
              rnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, rnn.loss, rnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

            return loss,accuracy

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              rnn.input_x: x_batch,
              rnn.input_y: y_batch,
              rnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, rnn.loss, rnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

            return accuracy

        # Create batches agnostic of class distributions
        batches = data_helpers.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

        # Create batches aware of imbalance in class distributions
        # batches = data_helpers.makeBatches(x_train, y_train[:,1].tolist(), FLAGS.batch_size, FLAGS.num_epochs)

        # Training loop. For each batch...
        prev_val_acc = 0
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_loss, train_acc = train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nTrain loss:{0}, Train accuracy:{1}".format(train_loss, train_acc))
                print("Evaluation:")
                val_acc = dev_step(x_dev, y_dev, writer=dev_summary_writer)
                if val_acc > 0.94 and val_acc > prev_val_acc:
                    save_path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Model checkpoint saved at {0}, accuracy={1}".format(save_path, round(val_acc, 3)))
                    prev_val_acc = val_acc

                print("")