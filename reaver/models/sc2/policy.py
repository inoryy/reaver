import gin
import tensorflow as tf
from reaver.models.base import MultiPolicy


@gin.configurable
class SC2MultiPolicy(MultiPolicy):
    def __init__(self, act_spec, logits):
        super().__init__(act_spec, logits)

        args_mask = tf.constant(act_spec.spaces[0].args_mask, dtype=tf.float32)
        act_args_mask = tf.gather(args_mask, self.inputs[0])
        act_args_mask = tf.transpose(act_args_mask, [1, 0])

        self.logli = self.dists[0].log_prob(self.inputs[0])
        for i in range(1, len(self.dists)):
            self.logli += act_args_mask[i-1] * self.dists[i].log_prob(self.inputs[i])
