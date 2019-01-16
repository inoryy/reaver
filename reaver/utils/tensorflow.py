import gin
import tensorflow as tf
gin.external_configurable(tf.train.AdamOptimizer, module='tf.train')
gin.external_configurable(tf.train.RMSPropOptimizer, module='tf.train')
gin.external_configurable(tf.train.get_global_step, module='tf.train')
gin.external_configurable(tf.train.piecewise_constant, module='tf.train')
gin.external_configurable(tf.train.polynomial_decay, module='tf.train')
gin.external_configurable(tf.initializers.orthogonal, 'tf.initializers.orthogonal')


class SessionManager:
    def __init__(self, sess=None, base_path='results/', checkpoint_freq=100, training_enabled=True):
        if not sess:
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        tf.keras.backend.set_session(sess)

        self.sess = sess
        self.saver = None
        self.base_path = base_path
        self.checkpoint_freq = checkpoint_freq
        self.training_enabled = training_enabled
        self.global_step = tf.train.get_or_create_global_step()
        self.summary_writer = tf.summary.FileWriter(self.summaries_path)

    def restore_or_init(self):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.latest_checkpoint(self.checkpoints_path)
        if checkpoint:
            self.saver.restore(self.sess, checkpoint)

            if self.training_enabled:
                # merge with previous summary session
                self.summary_writer.add_session_log(
                    tf.SessionLog(status=tf.SessionLog.START), self.sess.run(self.global_step))
        else:
            self.sess.run(tf.global_variables_initializer())
        # this call locks the computational graph into read-only state,
        # as a safety measure against memory leaks caused by mistakingly adding new ops to it
        self.sess.graph.finalize()

    def run(self, tf_op, tf_inputs, inputs):
        return self.sess.run(tf_op, feed_dict=dict(zip(tf_inputs, inputs)))

    def on_update(self, step):
        if not self.checkpoint_freq or not self.training_enabled or step % self.checkpoint_freq:
            return

        self.saver.save(self.sess, self.checkpoints_path + '/ckpt', global_step=step)

    def add_summaries(self, tags, values, prefix='', step=None):
        for tag, value in zip(tags, values):
            self.add_summary(tag, value, prefix, step)

    def add_summary(self, tag, value, prefix='', step=None):
        if not self.training_enabled:
            return
        summary = self.create_summary(prefix + '/' + tag, value)
        self.summary_writer.add_summary(summary, global_step=step)

    @staticmethod
    def create_summary(tag, value):
        return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])

    @property
    def start_step(self):
        if self.training_enabled:
            return self.global_step.eval(session=self.sess)
        return 0

    @property
    def summaries_path(self):
        return self.base_path + '/summaries'

    @property
    def checkpoints_path(self):
        return self.base_path + '/checkpoints'
