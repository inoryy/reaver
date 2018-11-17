import tensorflow as tf


class Saver:
    def __init__(self, sess, checkpoint_path, checkpoint_freq):
        self.sess = sess
        self.saver = None
        self.checkpoint_path = checkpoint_path
        self.checkpoint_freq = checkpoint_freq
        self.global_step = tf.train.get_or_create_global_step()

    def restore_or_init(self):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.latest_checkpoint(self.checkpoint_path)
        if checkpoint:
            self.saver.restore(self.sess, checkpoint)
        else:
            self.sess.run(tf.global_variables_initializer())

    def on_update(self, step):
        if step % self.checkpoint_freq:
            return
        self.saver.save(self.sess, self.checkpoint_path + '/ckpt', global_step=self.global_step)


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