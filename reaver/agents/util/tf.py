# https://github.com/tambetm/TSCL/blob/master/addition/tensorboard_utils.py
import tensorflow as tf

optimizers = dict(
    adam=tf.train.AdamOptimizer,
    rmsprop=tf.train.RMSPropOptimizer
)


def tf_run(sess, tf_op, tf_inputs, inputs):
    return sess.run(tf_op, feed_dict=dict(zip(tf_inputs, inputs)))


def create_summary_writer(logdir):
    return tf.summary.FileWriter(logdir)


def create_summary(tag, value):
    return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])


def add_summary(writer, tag, value, step):
    writer.add_summary(create_summary(tag, value), global_step=step)


def add_summaries(writer, tags, values, step, prefix=''):
    for (t, v) in zip(tags, values):
        s = create_summary(prefix + '/' + t, v)
        writer.add_summary(s, global_step=step)
