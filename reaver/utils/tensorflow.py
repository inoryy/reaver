import tensorflow as tf


def tf_run(sess, tf_op, tf_inputs, inputs):
    return sess.run(tf_op, feed_dict=dict(zip(tf_inputs, inputs)))


# https://github.com/tambetm/TSCL/blob/master/addition/tensorboard_utils.py
def create_summary_writer(logdir):
    return tf.summary.FileWriter(logdir)


def create_summary(tag, value):
    return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])


def add_summary(writer, tag, value, step, prefix=''):
    if not writer:
        return
    writer.add_summary(create_summary(prefix + '/' + tag, value), global_step=step)


def add_summaries(writer, tags, values, step, prefix=''):
    for (t, v) in zip(tags, values):
        add_summary(writer, t, v, step, prefix)